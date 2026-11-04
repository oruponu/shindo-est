# Command line

The `shindo-est` command packs and inspects model files. It also loads a
model file and answers prediction requests over standard input and standard
output. This document describes the commands and the request and answer
formats for programs written in any language that invoke the command.

Build it from the repository root with:

```sh
cargo build --release -p shindo-est-cli
```

The command is placed at `target/release/shindo-est` (`shindo-est.exe` on
Windows).

## Commands

```text
shindo-est info <model>
shindo-est pack <manifest> <classifier> <regressor> <kuril> <izu_bonin> <out>
shindo-est predict --model <model> [--max-threads <n>]
shindo-est --help | --version
```

- `info` prints the manifest of a model file.
- `pack` packs a manifest, the classifier and the regressor in LightGBM's text
  format, and the Kuril and Izu-Bonin grids as `lon,lat,depth,thickness` CSV
  files into a model file (see [model-format.md](model-format.md)). It checks
  that the packed bytes can be read as a model file before writing them, and
  replaces `<out>` only after the new file has been written, read back, and
  verified to match the packed bytes.
- `predict` loads the model file once and answers requests, one per line,
  until standard input reaches EOF. `--max-threads` sets the maximum number of
  threads used to process each request. It accepts an integer of at least 1.
  The default is the number of logical CPUs. The thread limit does not affect
  the answers.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. For `predict`, standard input reached EOF or standard output was closed. |
| 1 | The command failed. Standard error contains the error and its causes, starting with `shindo-est: `. |
| 2 | The command-line arguments are invalid. Standard error contains the error and usage information. |

## Requests and answers of `predict`

Each nonblank line of standard input is treated as a request, which must be
a JSON object encoded in UTF-8. Each request gets exactly one answer, a JSON
object on one line of standard output, in the order of the requests.
Standard output is flushed after each answer, so a program can keep the
command running and send a request whenever it needs one.

- A line ends with LF or with CRLF, and the last line may have no line break.
- A line containing only JSON whitespace (spaces, tabs, and CR) is skipped.
  No answer is produced.
- A line longer than 16 MiB (16,777,216 bytes, excluding the line break)
  produces a `request_too_large` error answer. The next line is processed as
  usual. A request for 4,400 sites is typically 0.2 to 0.3 MB.
- An invalid line gets an error answer, and the next line is processed as
  usual.

### Requests

```json
{"id": 7, "hypocenter": {"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}, "sites": [{"latitude": 35.02, "longitude": 140.02, "arv": 1.0}, {"latitude": 35.05, "longitude": 140.05, "arv": 0.0}], "options": {"offset": 0.2}}
```

| Key | Value |
|---|---|
| `id` | Optional. Any JSON value, which the answer repeats as written. |
| `hypocenter.latitude`, `hypocenter.longitude` | Degrees. |
| `hypocenter.depth` | km. `null` if unknown. |
| `hypocenter.magnitude` | JMA magnitude (Mj). `null` if unknown. |
| `sites` | An array of sites, which may be empty. |
| `sites[].latitude`, `sites[].longitude` | Degrees. |
| `sites[].arv` | Amplification of peak ground velocity relative to engineering bedrock. |
| `options` | Optional. |
| `options.offset` | Optional, 0 by default. An amount added to the instrumental intensity. |

All listed keys in `hypocenter` and each site are required. Unknown keys are
not allowed in the request, `hypocenter`, `options`, or site objects; the
value of `id` may be any JSON. The request, `hypocenter`, and each entry in
`sites` must be JSON objects, not arrays of values, as must `options` when
present.

### Numbers

- In numeric fields, `null` represents NaN (an unknown value). Whether NaN is
  valid depends on the field, as described in the following sections.
- Numbers in requests, except those in `id`, are rounded to the nearest
  IEEE 754 binary64 value. Finite binary64 values in answers are written as
  decimals that round-trip to the same values. A number too large for
  binary64, such as `1e400`, is the error `invalid_json`.
- A number in an answer that has no finite value is written as `null`.
- The `id` value is not converted to binary64. It is repeated as written, so
  values such as `18446744073709551617` and `1.0e+2` are preserved.

### Answers

The answer to the request above:

```json
{"id":7,"model_id":"synthetic-fixture","offset":0.2,"results":[{"status":"ok","class":"4","instrumental":4.1000000000000005,"probability":0.9308615796566531,"regression":3.9000000000000004},{"status":"invalid_input","reason":"invalid_arv","class":null,"instrumental":null,"probability":null,"regression":null}]}
```

An answer is written as a single-line JSON object without added formatting
whitespace. It includes `id` only if the request has one, followed by
`model_id` (from the model's manifest), `offset`, and `results`. The `results`
array contains one entry per site, in the same order as `sites`. Each entry
has these keys, in this order:

| Key | Value |
|---|---|
| `status` | `ok`, `out_of_scope`, or `invalid_input`. |
| `reason` | Only when `status` is not `ok`: why the site got no intensity (see below). |
| `class` | The JMA seismic intensity class of `instrumental`: `"0"` to `"4"`, `"5-"`, `"5+"`, `"6-"`, `"6+"`, or `"7"`. `null` without an intensity. |
| `instrumental` | The instrumental intensity, with the offset. |
| `probability` | The probability that the site is felt, with the offset applied. |
| `regression` | The regressor's instrumental intensity before the offset, for diagnosis. |

If `status` is not `ok`, `class`, `instrumental`, `probability`, and
`regression` are all `null`.

| `status` | `reason` | When |
|---|---|---|
| `invalid_input` | `invalid_latitude` | The site's latitude is `null` or not within -90 to 90. |
| `invalid_input` | `invalid_longitude` | The site's longitude is `null` or not within -180 to 180. |
| `invalid_input` | `invalid_arv` | The site's ARV is `null` or not positive. |
| `out_of_scope` | `unknown_depth` | The depth is `null`. |
| `out_of_scope` | `unknown_magnitude` | The magnitude is `null`. |
| `out_of_scope` | `non_positive_magnitude` | The magnitude is at or below 0. |
| `out_of_scope` | `out_of_scope` | Another reason, added in a later version. |

A site with an invalid latitude, longitude, or ARV is `invalid_input` even
when the request is out of scope. All other sites in an out-of-scope request
have status `out_of_scope` with the same reason: the first applicable one
among `unknown_depth`, `unknown_magnitude`, and `non_positive_magnitude`, in
that order.

### Error answers

```json
{"id":7,"error":{"code":"invalid_hypocenter_depth","message":"the hypocenter depth must be finite and at least 0, or NaN if unknown"}}
```

The answer has `id` if the line is a JSON object containing an `id`, with
two exceptions. A `request_too_large` answer has no `id`, because oversized
lines are not parsed as JSON. If the JSON text of an `id` contains a raw
carriage return between tokens, as in `[1,<CR>2]`, the request is rejected
with `invalid_request`, and the answer omits `id` to keep the answer on a
single line. (A raw carriage return inside a JSON string is not valid JSON
and gives `invalid_json`.)
`message` is an English description for people; programs should look at
`code`.

| `code` | When |
|---|---|
| `invalid_json` | The line is not JSON: not UTF-8, a syntax error, or a number out of range. |
| `invalid_request` | The line is valid JSON but not a valid request: a non-object value, a missing or unknown key, a value of the wrong type, or a raw carriage return between tokens in the JSON text of `id`. |
| `request_too_large` | The line is longer than 16 MiB. |
| `invalid_hypocenter_latitude` | The hypocenter's latitude is `null` or not within -90 to 90. |
| `invalid_hypocenter_longitude` | The hypocenter's longitude is `null` or not within -180 to 180. |
| `invalid_hypocenter_depth` | The depth is negative. |
| `invalid_hypocenter_magnitude` | The magnitude is positive infinity. A request cannot give it, as `1e400` is out of range, but the code is reserved so that the codes match the library's errors. |
| `invalid_offset` | The offset is `null`. |
| `invalid_input` | Another invalid input, added in a later version. |

### Standard error

Standard error is a log for people, not part of the protocol. `predict`
writes `shindo-est: loaded model <model_id> from <path>` after loading the
model file, and `shindo-est: line <n>: <code>: <message>` for each error
answer, where `<n>` counts the lines of standard input from 1, including
skipped ones. Control characters in the message, such as a line break from a
key of the request, are escaped as `\n` and the like, so that each entry
stays on one line. A failure to write standard error does not stop the
answers.

If standard output is closed, `predict` stops and exits with code 0. If
reading from standard input or writing to standard output fails for any
other reason, it exits with code 1.
