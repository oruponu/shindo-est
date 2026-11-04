//! The requests and answers of the `predict` command, one JSON object a line.
//!
//! A number of a request may be `null`, which means NaN, and a number of an
//! answer that is not finite is written as `null`. Numbers are read with
//! `float_roundtrip`, so that the decimal text and the `f64` match bit for bit.

use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use serde::de::value::MapAccessDeserializer;
use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::error::Category;
use serde_json::value::RawValue;
use shindo_est::validate::{self, HypocenterField, OutOfScopeReason, Scope, SiteField};
use shindo_est::{
    Hypocenter, InputError, IntensityClass, Options, PointResult, PointStatus, Predictor, Site,
};

/// The answer to one line.
#[derive(Debug)]
pub(crate) struct Answer {
    /// The answer as JSON, without a line break.
    pub json: String,
    /// For an error answer, "<code>: <message>" for the log, with control
    /// characters escaped so that it stays on one line.
    pub error: Option<String>,
}

/// Answers the request on `line` with `predictor`, using at most
/// `max_threads` threads, or as many as there are logical CPUs for `None`.
pub(crate) fn answer(
    predictor: &Predictor,
    model_id: &str,
    max_threads: Option<NonZeroUsize>,
    line: &[u8],
) -> Answer {
    let request = match read(line) {
        Ok(request) => request,
        Err(rejection) => {
            return error_answer(rejection.id, rejection.code, &rejection.message);
        }
    };
    let input_error =
        |error: InputError| error_answer(request.id, input_error_code(error), &error.to_string());
    let scope = match validate::check_call(&request.hypocenter, request.offset) {
        Ok(scope) => scope,
        Err(error) => return input_error(error),
    };
    let mut options = Options::default().with_offset(request.offset);
    if let Some(threads) = max_threads {
        options = options.with_max_threads(threads);
    }
    let points = match predictor.predict(&request.hypocenter, &request.sites, &options) {
        Ok(points) => points,
        Err(error) => return input_error(error),
    };
    let mut results = Vec::with_capacity(points.len());
    let sites = points.iter().zip(&request.sites);
    for (point, site) in sites {
        results.push(site_answer(point, site, scope));
    }
    let success = SuccessAnswer {
        id: request.id,
        model_id,
        offset: request.offset,
        results,
    };
    Answer {
        json: to_json(&success),
        error: None,
    }
}

pub(crate) fn error_answer(id: Option<&RawValue>, code: &str, message: &str) -> Answer {
    let answer = ErrorAnswer {
        id,
        error: ErrorBody { code, message },
    };
    Answer {
        json: to_json(&answer),
        error: Some(format!("{code}: {}", escape_control(message))),
    }
}

/// Returns `text` with each control character escaped as in a Rust string,
/// such as a line break as "\n". A message can hold one from a request, as
/// serde names an unknown key as decoded.
fn escape_control(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        if character.is_control() {
            escaped.extend(character.escape_default());
        } else {
            escaped.push(character);
        }
    }
    escaped
}

struct Request<'line> {
    /// The ID as written in the request, if it has one.
    id: Option<&'line RawValue>,
    hypocenter: Hypocenter,
    sites: Vec<Site>,
    offset: f64,
}

/// Why a line cannot be answered with predictions.
struct Rejection<'line> {
    id: Option<&'line RawValue>,
    code: &'static str,
    message: String,
}

/// Reads the request on `line`, in three steps: the line must be JSON, its
/// ID is taken if it is an object with one, and then it must be a request.
fn read(line: &[u8]) -> Result<Request<'_>, Rejection<'_>> {
    let json: Result<&RawValue, _> = serde_json::from_slice(line);
    let text = match json {
        Ok(raw) => raw.get(),
        Err(error) => {
            return Err(Rejection {
                id: None,
                code: "invalid_json",
                message: error.to_string(),
            });
        }
    };
    let id_only: Result<JsonObject<IdOnly<'_>>, _> = serde_json::from_str(text);
    let id = id_only.ok().and_then(|object| object.0.id);
    // A raw CR can only be whitespace between the tokens of the ID, as JSON
    // strings must escape it, but an answer that repeats it would be split by
    // a caller that reads lines with universal newlines.
    if id.is_some_and(|raw| raw.get().contains('\r')) {
        return Err(Rejection {
            id: None,
            code: "invalid_request",
            message: "the id has a raw carriage return between its tokens".to_owned(),
        });
    }
    let body: Result<JsonObject<RequestBody>, _> = serde_json::from_str(text);
    let body = match body {
        Ok(body) => body.0,
        Err(error) => {
            // A number out of the range of f64, which the first step lets
            // through, is a syntax error here.
            let code = match error.classify() {
                Category::Data => "invalid_request",
                Category::Syntax | Category::Eof | Category::Io => "invalid_json",
            };
            return Err(Rejection {
                id,
                code,
                message: error.to_string(),
            });
        }
    };
    let hypocenter = body.hypocenter.0;
    let mut sites = Vec::with_capacity(body.sites.len());
    for site in body.sites {
        let site = site.0;
        sites.push(Site {
            latitude: site.latitude,
            longitude: site.longitude,
            arv: site.arv,
        });
    }
    Ok(Request {
        id,
        hypocenter: Hypocenter {
            latitude: hypocenter.latitude,
            longitude: hypocenter.longitude,
            depth: hypocenter.depth,
            magnitude: hypocenter.magnitude,
        },
        sites,
        offset: body.options.0.offset,
    })
}

/// The ID of a request, read on its own so that an error answer can carry it
/// whatever else is wrong with the request.
#[derive(Deserialize)]
struct IdOnly<'line> {
    #[serde(borrow, default, deserialize_with = "raw_value")]
    id: Option<&'line RawValue>,
}

/// Reads a value as written, keeping `null` as a value.
fn raw_value<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<&'de RawValue>, D::Error> {
    let raw: &RawValue = Deserialize::deserialize(deserializer)?;
    Ok(Some(raw))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestBody {
    /// Read by [`IdOnly`] instead.
    #[serde(default, rename = "id")]
    _id: IgnoredAny,
    hypocenter: JsonObject<HypocenterBody>,
    sites: Vec<JsonObject<SiteBody>>,
    #[serde(default)]
    options: JsonObject<OptionsBody>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HypocenterBody {
    #[serde(deserialize_with = "number_or_null")]
    latitude: f64,
    #[serde(deserialize_with = "number_or_null")]
    longitude: f64,
    #[serde(deserialize_with = "number_or_null")]
    depth: f64,
    #[serde(deserialize_with = "number_or_null")]
    magnitude: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteBody {
    #[serde(deserialize_with = "number_or_null")]
    latitude: f64,
    #[serde(deserialize_with = "number_or_null")]
    longitude: f64,
    #[serde(deserialize_with = "number_or_null")]
    arv: f64,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionsBody {
    #[serde(default, deserialize_with = "number_or_null")]
    offset: f64,
}

/// Reads a number, or `null` as NaN. A field read with it is still required,
/// as serde reports a missing field that has `deserialize_with`.
fn number_or_null<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let number: Option<f64> = Deserialize::deserialize(deserializer)?;
    Ok(number.unwrap_or(f64::NAN))
}

/// A value that must be a JSON object. Serde's derived structs also take an
/// array of their fields in order, which requests do not allow.
#[derive(Default)]
struct JsonObject<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for JsonObject<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer
            .deserialize_map(ObjectVisitor(PhantomData))
            .map(JsonObject)
    }
}

struct ObjectVisitor<T>(PhantomData<T>);

impl<'de, T: Deserialize<'de>> Visitor<'de> for ObjectVisitor<T> {
    type Value = T;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<T, A::Error> {
        T::deserialize(MapAccessDeserializer::new(map))
    }
}

#[derive(Serialize)]
struct SuccessAnswer<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a RawValue>,
    model_id: &'a str,
    offset: f64,
    results: Vec<SiteAnswer>,
}

/// The answer for one site. The fields are serialized in this order.
#[derive(Serialize)]
struct SiteAnswer {
    status: &'static str,
    /// Why the site got no intensity, for a status other than "ok".
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
    class: Option<&'static str>,
    instrumental: Option<f64>,
    probability: Option<f64>,
    regression: Option<f64>,
}

#[derive(Serialize)]
struct ErrorAnswer<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<&'a RawValue>,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    message: &'a str,
}

/// Returns the answer for `site`, which got `point` in a call of `scope`.
fn site_answer(point: &PointResult, site: &Site, scope: Scope) -> SiteAnswer {
    let (status, reason) = match point.status {
        PointStatus::Ok => ("ok", None),
        PointStatus::OutOfScope => {
            let reason = match scope {
                Scope::OutOfScope(reason) => out_of_scope_reason(reason),
                Scope::InScope => "out_of_scope",
            };
            ("out_of_scope", Some(reason))
        }
        // A status the core adds later also means the site got no intensity.
        _ => {
            let field = validate::check_site(site).err();
            (
                "invalid_input",
                Some(field.map_or("invalid_input", site_reason)),
            )
        }
    };
    let class = IntensityClass::from_instrumental(point.instrumental);
    SiteAnswer {
        status,
        reason,
        class: class.map(IntensityClass::label),
        instrumental: finite(point.instrumental),
        probability: finite(point.probability),
        regression: finite(point.regression),
    }
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn out_of_scope_reason(reason: OutOfScopeReason) -> &'static str {
    match reason {
        OutOfScopeReason::UnknownDepth => "unknown_depth",
        OutOfScopeReason::UnknownMagnitude => "unknown_magnitude",
        OutOfScopeReason::NonPositiveMagnitude => "non_positive_magnitude",
        // A reason the core adds later.
        _ => "out_of_scope",
    }
}

fn site_reason(field: SiteField) -> &'static str {
    match field {
        SiteField::Latitude => "invalid_latitude",
        SiteField::Longitude => "invalid_longitude",
        SiteField::Arv => "invalid_arv",
    }
}

fn input_error_code(error: InputError) -> &'static str {
    match error {
        InputError::InvalidHypocenter(HypocenterField::Latitude) => "invalid_hypocenter_latitude",
        InputError::InvalidHypocenter(HypocenterField::Longitude) => "invalid_hypocenter_longitude",
        InputError::InvalidHypocenter(HypocenterField::Depth) => "invalid_hypocenter_depth",
        InputError::InvalidHypocenter(HypocenterField::Magnitude) => "invalid_hypocenter_magnitude",
        InputError::InvalidOffset => "invalid_offset",
        // An error the core adds later.
        _ => "invalid_input",
    }
}

fn to_json(answer: &impl Serialize) -> String {
    // The answers have no maps with keys other than strings, which is the
    // only way they could fail to serialize.
    serde_json::to_string(answer).expect("answers serialize")
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::Value;
    use shindo_est::validate::{HypocenterField, Scope};
    use shindo_est::{
        Hypocenter, InputError, IntensityClass, Options, PointResult, PointStatus, Site,
    };

    use crate::test_support;

    /// The hypocenter of the model file fixture's tests, as JSON.
    const HYPOCENTER: &str =
        r#"{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}"#;

    /// The sites of the model file fixture's tests, as JSON: five valid sites
    /// and one with an invalid ARV.
    const SITES: &str = r#"[
        {"latitude": 35.02, "longitude": 140.02, "arv": 1.0},
        {"latitude": 35.02, "longitude": 140.17, "arv": 2.0},
        {"latitude": 35.12, "longitude": 140.12, "arv": 1.0},
        {"latitude": 34.9, "longitude": 140.1, "arv": 1.0},
        {"latitude": 40.0, "longitude": 135.0, "arv": 1.0},
        {"latitude": 35.02, "longitude": 140.02, "arv": 0.0}
    ]"#;

    fn answer(line: &[u8]) -> super::Answer {
        let bundle = test_support::bundle();
        super::answer(bundle.predictor(), test_support::MODEL_ID, None, line)
    }

    fn answer_value(line: &str) -> Value {
        let answered = answer(line.as_bytes());
        serde_json::from_str(&answered.json).expect("the answer is JSON")
    }

    /// Returns the error code of the answer to `line`, failing if it is not an
    /// error.
    fn error_code(line: &str) -> String {
        let value = answer_value(line);
        let code = value["error"]["code"].as_str();
        code.unwrap_or_else(|| panic!("{line} got {value}"))
            .to_owned()
    }

    /// Returns a request with ID 7 for `hypocenter` and `sites`.
    fn request(hypocenter: &str, sites: &str) -> String {
        format!(r#"{{"id": 7, "hypocenter": {hypocenter}, "sites": {sites}}}"#)
    }

    /// Returns "<status>: <reason>" for each result of the answer, or only
    /// the status for a result without a reason.
    fn statuses(value: &Value) -> Vec<String> {
        let results = value["results"].as_array();
        let results = results.unwrap_or_else(|| panic!("no results in {value}"));
        let mut statuses = Vec::new();
        for result in results {
            let status = result["status"].as_str().unwrap_or_default();
            let status = match result["reason"].as_str() {
                Some(reason) => format!("{status}: {reason}"),
                None => status.to_owned(),
            };
            statuses.push(status);
        }
        statuses
    }

    /// Returns a number of an answer, with `null` as NaN.
    fn number(value: &Value) -> f64 {
        value.as_f64().unwrap_or(f64::NAN)
    }

    fn same_bits(left: f64, right: f64) -> bool {
        left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
    }

    #[test]
    fn answers_a_request_like_the_library() {
        let line = format!(
            r#"{{"id": 7, "hypocenter": {HYPOCENTER}, "sites": {SITES}, "options": {{"offset": 0.2}}}}"#
        );
        let hypocenter = Hypocenter {
            latitude: 35.02,
            longitude: 140.02,
            depth: 80.0,
            magnitude: 6.5,
        };
        let sites: Vec<Site> = test_support::SITES.to_vec();
        let bundle = test_support::bundle();
        let options = Options::default().with_offset(0.2);
        let points = bundle.predictor().predict(&hypocenter, &sites, &options);
        let points = points.expect("the fixture's call is valid");

        let value = answer_value(&line);

        assert_eq!(value["model_id"], test_support::MODEL_ID, "{value}");
        assert!(same_bits(number(&value["offset"]), 0.2), "{value}");
        let results = value["results"].as_array().expect("the answer has results");
        assert_eq!(results.len(), points.len(), "{value}");
        let mut statuses = Vec::new();
        let pairs = results.iter().zip(&points);
        for (result, point) in pairs {
            let status = match point.status {
                PointStatus::Ok => "ok",
                PointStatus::OutOfScope => "out_of_scope",
                _ => "invalid_input",
            };
            assert_eq!(result["status"], status, "{result}");
            statuses.push(status);
            let class = IntensityClass::from_instrumental(point.instrumental);
            assert_eq!(result["class"].as_str(), class.map(IntensityClass::label));
            for (key, expected) in [
                ("instrumental", point.instrumental),
                ("probability", point.probability),
                ("regression", point.regression),
            ] {
                let got = number(&result[key]);
                assert!(same_bits(got, expected), "{key}: {got} and {expected}");
            }
        }
        assert!(statuses.contains(&"ok"), "{statuses:?}");
        assert!(statuses.contains(&"invalid_input"), "{statuses:?}");
    }

    #[test]
    fn echoes_the_id_as_given() {
        for id in [
            "7",
            r#""a""#,
            // An escaped CR, two characters that the answer keeps as such.
            r#""a\rb""#,
            r#"{"k": [1]}"#,
            "null",
            "18446744073709551617",
            "1.0e+2",
            "-0.0",
        ] {
            let line = format!(r#"{{"id": {id}, "hypocenter": {HYPOCENTER}, "sites": []}}"#);
            let answered = answer(line.as_bytes()).json;
            let start = format!(r#"{{"id":{id},"model_id":"#);
            assert!(answered.starts_with(&start), "{answered}");

            let line = format!(r#"{{"id": {id}, "sites": []}}"#);
            let failed = answer(line.as_bytes()).json;
            let start = format!(r#"{{"id":{id},"error":"#);
            assert!(failed.starts_with(&start), "{failed}");
        }

        let line = format!(r#"{{"hypocenter": {HYPOCENTER}, "sites": []}}"#);
        let answered = answer(line.as_bytes()).json;
        assert!(answered.starts_with(r#"{"model_id":"#), "{answered}");
        let failed = answer(br#"{"sites": []}"#).json;
        assert!(failed.starts_with(r#"{"error":"#), "{failed}");
    }

    #[test]
    fn null_numbers_mean_nan() {
        let unknown_depth =
            r#"{"latitude": 35.02, "longitude": 140.02, "depth": null, "magnitude": 6.5}"#;
        let value = answer_value(&request(unknown_depth, SITES));
        let mut expected = vec!["out_of_scope: unknown_depth"; 5];
        expected.push("invalid_input: invalid_arv");
        assert_eq!(statuses(&value), expected, "{value}");

        let unknown_magnitude =
            r#"{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": null}"#;
        let value = answer_value(&request(unknown_magnitude, SITES));
        assert_eq!(
            statuses(&value)[0],
            "out_of_scope: unknown_magnitude",
            "{value}"
        );

        let no_magnitude =
            r#"{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 0}"#;
        let value = answer_value(&request(no_magnitude, SITES));
        assert_eq!(
            statuses(&value)[0],
            "out_of_scope: non_positive_magnitude",
            "{value}"
        );

        let no_arv = r#"[{"latitude": 35.02, "longitude": 140.02, "arv": null}]"#;
        let value = answer_value(&request(HYPOCENTER, no_arv));
        assert_eq!(statuses(&value), ["invalid_input: invalid_arv"], "{value}");

        let no_latitude =
            r#"{"latitude": null, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}"#;
        assert_eq!(
            error_code(&request(no_latitude, SITES)),
            "invalid_hypocenter_latitude"
        );
    }

    #[test]
    fn writes_null_and_a_reason_exactly() {
        let line = r#"{"id": 1, "hypocenter": {"latitude": 35.02, "longitude": 140.02, "depth": null, "magnitude": 6.5}, "sites": [{"latitude": 35.02, "longitude": 140.02, "arv": 1.0}, {"latitude": 35.02, "longitude": 140.02, "arv": 0.0}]}"#;

        let answered = answer(line.as_bytes());

        assert_eq!(
            answered.json,
            concat!(
                r#"{"id":1,"model_id":"synthetic-fixture","offset":0.0,"results":["#,
                r#"{"status":"out_of_scope","reason":"unknown_depth","class":null,"#,
                r#""instrumental":null,"probability":null,"regression":null},"#,
                r#"{"status":"invalid_input","reason":"invalid_arv","class":null,"#,
                r#""instrumental":null,"probability":null,"regression":null}]}"#
            )
        );
        assert_eq!(answered.error, None);

        let mut computed = PointResult::default();
        computed.status = PointStatus::Ok;
        computed.instrumental = 4.5;
        computed.probability = 0.1;
        computed.regression = 1e-7;
        let site = test_support::SITES[0];
        let written = serde_json::to_string(&super::site_answer(&computed, &site, Scope::InScope));
        assert_eq!(
            written.expect("a site answer serializes"),
            r#"{"status":"ok","class":"5-","instrumental":4.5,"probability":0.1,"regression":1e-7}"#
        );
    }

    #[test]
    fn rejects_missing_unknown_and_mistyped_fields() {
        let without_depth = r#"{"latitude": 35.02, "longitude": 140.02, "magnitude": 6.5}"#;
        let misspelled =
            r#"{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitde": 6.5}"#;
        let extra = r#"{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5, "foo": 1}"#;
        let text_depth =
            r#"{"latitude": 35.02, "longitude": 140.02, "depth": "10", "magnitude": 6.5}"#;
        let site = r#"{"latitude": 35.02, "longitude": 140.02, "arv": 1.0}"#;
        let extra_site = r#"[{"latitude": 35.02, "longitude": 140.02, "arv": 1.0, "vs30": 400}]"#;
        let lines = [
            request(without_depth, SITES),
            request(misspelled, SITES),
            request(extra, SITES),
            request(text_depth, SITES),
            request(HYPOCENTER, site),
            request(HYPOCENTER, extra_site),
            request("[35.02, 140.02, 80.0, 6.5]", SITES),
            request(HYPOCENTER, "[[35.02, 140.02, 1.0]]"),
            format!(r#"{{"id": 7, "hypocenter": {HYPOCENTER}, "sites": [], "foo": 1}}"#),
            format!(
                r#"{{"id": 7, "hypocenter": {HYPOCENTER}, "sites": [], "options": {{"offst": 0.2}}}}"#
            ),
            format!(r#"{{"id": 7, "hypocenter": {HYPOCENTER}, "sites": [], "options": [0.2]}}"#),
        ];
        for line in &lines {
            let value = answer_value(line);
            assert_eq!(value["error"]["code"], "invalid_request", "{line}: {value}");
            assert_eq!(value["id"], 7, "{line}: {value}");
            let message = value["error"]["message"].as_str().unwrap_or_default();
            assert!(!message.is_empty(), "{line}: {value}");
        }

        for line in ["[7]", "5", r#""request""#, "null"] {
            let answered = answer(line.as_bytes()).json;
            assert!(
                answered.starts_with(r#"{"error":{"code":"invalid_request","#),
                "{line}: {answered}"
            );
        }
    }

    #[test]
    fn options_and_offset_are_optional() {
        for line in [
            format!(r#"{{"hypocenter": {HYPOCENTER}, "sites": []}}"#),
            format!(r#"{{"hypocenter": {HYPOCENTER}, "sites": [], "options": {{}}}}"#),
        ] {
            let answered = answer(line.as_bytes()).json;
            assert_eq!(
                answered, r#"{"model_id":"synthetic-fixture","offset":0.0,"results":[]}"#,
                "{line}"
            );
        }
    }

    #[test]
    fn reports_input_errors_with_their_codes() {
        let cases = [
            (
                r#"{"latitude": 91, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}"#,
                "",
                InputError::InvalidHypocenter(HypocenterField::Latitude),
                "invalid_hypocenter_latitude",
            ),
            (
                r#"{"latitude": 35.02, "longitude": 181, "depth": 80.0, "magnitude": 6.5}"#,
                "",
                InputError::InvalidHypocenter(HypocenterField::Longitude),
                "invalid_hypocenter_longitude",
            ),
            (
                r#"{"latitude": 35.02, "longitude": 140.02, "depth": -1, "magnitude": 6.5}"#,
                "",
                InputError::InvalidHypocenter(HypocenterField::Depth),
                "invalid_hypocenter_depth",
            ),
            (
                HYPOCENTER,
                r#", "options": {"offset": null}"#,
                InputError::InvalidOffset,
                "invalid_offset",
            ),
        ];
        for (hypocenter, options, error, code) in cases {
            let line =
                format!(r#"{{"id": 7, "hypocenter": {hypocenter}, "sites": {SITES}{options}}}"#);

            let answered = answer(line.as_bytes());

            let value: Value = serde_json::from_str(&answered.json).expect("the answer is JSON");
            assert_eq!(value["error"]["code"], code, "{line}");
            assert_eq!(value["error"]["message"], error.to_string(), "{line}");
            assert_eq!(answered.error, Some(format!("{code}: {error}")));
        }
    }

    #[test]
    fn every_input_error_has_a_code() {
        for (error, code) in [
            (
                InputError::InvalidHypocenter(HypocenterField::Latitude),
                "invalid_hypocenter_latitude",
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Longitude),
                "invalid_hypocenter_longitude",
            ),
            (
                InputError::InvalidHypocenter(HypocenterField::Depth),
                "invalid_hypocenter_depth",
            ),
            // No request can reach this error: JSON has no positive
            // infinity, and 1e400 is out of range, so only this test covers it.
            (
                InputError::InvalidHypocenter(HypocenterField::Magnitude),
                "invalid_hypocenter_magnitude",
            ),
            (InputError::InvalidOffset, "invalid_offset"),
        ] {
            assert_eq!(super::input_error_code(error), code, "{error:?}");
        }
    }

    #[test]
    fn rejects_an_id_with_a_raw_carriage_return() {
        let line = format!("{{\"id\": [1,\r2], \"hypocenter\": {HYPOCENTER}, \"sites\": []}}");

        let answered = answer(line.as_bytes()).json;

        assert!(!answered.contains('\r'), "{answered:?}");
        assert!(
            answered.starts_with(r#"{"error":{"code":"invalid_request","#),
            "{answered:?}"
        );
    }

    #[test]
    fn accepts_a_raw_carriage_return_outside_the_id() {
        // The answer does not repeat the other parts of the request.
        let line = format!("{{\"id\": 1,\r\"hypocenter\": {HYPOCENTER},\r\"sites\": []}}");

        let answered = answer(line.as_bytes()).json;

        assert!(
            answered.starts_with(r#"{"id":1,"model_id":"#),
            "{answered:?}"
        );
    }

    #[test]
    fn errors_stay_on_one_line() {
        // The key has an escaped line break, which serde's message decodes.
        let answered = answer(br#"{"id": 7, "foo\nbar": 1}"#);

        assert!(!answered.json.contains('\n'), "{}", answered.json);
        let error = answered.error.expect("the answer is an error");
        assert!(!error.contains('\n'), "{error}");
        assert!(error.contains(r"foo\nbar"), "{error}");
    }

    #[test]
    fn invalid_json_has_no_id() {
        for line in [&br#"{"id": 7,"#[..], b"\xff", b"{} {}", b""] {
            let answered = answer(line).json;
            assert!(
                answered.starts_with(r#"{"error":{"code":"invalid_json","#),
                "{answered}"
            );
        }
    }

    #[test]
    fn out_of_range_numbers_are_invalid_json() {
        let latitude =
            r#"{"latitude": 1e400, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}"#;
        let arv = r#"[{"latitude": 35.02, "longitude": 140.02, "arv": 1e400}]"#;
        let lines = [
            request(latitude, SITES),
            request(HYPOCENTER, arv),
            format!(
                r#"{{"id": 7, "hypocenter": {HYPOCENTER}, "sites": [], "options": {{"offset": 1e400}}}}"#
            ),
        ];
        for line in &lines {
            let value = answer_value(line);
            assert_eq!(value["error"]["code"], "invalid_json", "{line}: {value}");
            assert_eq!(value["id"], 7, "{line}: {value}");
        }

        let line = format!(r#"{{"id": 1e400, "hypocenter": {HYPOCENTER}, "sites": []}}"#);
        let answered = answer(line.as_bytes()).json;
        assert!(
            answered.starts_with(r#"{"id":1e400,"model_id":"#),
            "{answered}"
        );
    }

    #[derive(Deserialize)]
    struct RequestNumber(#[serde(deserialize_with = "super::number_or_null")] f64);

    /// Returns the next number of the `SplitMix64` sequence.
    fn split_mix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn numbers_round_trip_bit_for_bit() {
        let mut state = 0;
        let mut checked = 0;
        while checked < 10_000 {
            let value = f64::from_bits(split_mix64(&mut state));
            if !value.is_finite() {
                continue;
            }
            let text = serde_json::to_string(&Some(value)).expect("a finite number serializes");

            let read: RequestNumber = serde_json::from_str(&text).expect("the number reads back");

            assert_eq!(read.0.to_bits(), value.to_bits(), "{text}");
            checked += 1;
        }
    }
}
