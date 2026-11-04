//! The `predict` command: answers requests, one line at a time, until the
//! input ends.

use std::error::Error;
use std::io::{self, BufRead, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use shindo_est::ModelBundle;

use crate::{info, protocol};

/// Longest request line in bytes, without its line break.
pub(crate) const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

/// Reads the model file at `model` and answers the requests on standard input
/// on standard output until the input ends, with the log on standard error.
pub(crate) fn run(model: &Path, max_threads: Option<NonZeroUsize>) -> Result<(), Box<dyn Error>> {
    let bundle = info::load(model)?;
    let mut log = io::stderr().lock();
    // The log is for people watching, so a failure to write it does not stop
    // the answers.
    let _ = writeln!(
        log,
        "shindo-est: loaded model {} from {}",
        bundle.manifest().model_id(),
        model.display()
    );
    let output = BufWriter::new(io::stdout().lock());
    serve(
        &bundle,
        max_threads,
        io::stdin().lock(),
        output,
        log,
        MAX_REQUEST_BYTES,
    )?;
    Ok(())
}

/// Answers each line of `input` with one line on `output`, which is flushed
/// after each answer, and writes a line on `log` for each error answer.
///
/// Lines of JSON whitespace only are skipped, and a line longer than
/// `max_request_bytes` is answered with an error without being kept in
/// memory.
///
/// # Errors
///
/// Returns an error when reading `input` or writing `output` fails, except
/// that a closed `output` ends the answers without an error, as the caller
/// stopped reading them.
pub(crate) fn serve(
    bundle: &ModelBundle,
    max_threads: Option<NonZeroUsize>,
    mut input: impl BufRead,
    mut output: impl Write,
    mut log: impl Write,
    max_request_bytes: usize,
) -> io::Result<()> {
    let predictor = bundle.predictor();
    let model_id = bundle.manifest().model_id();
    let mut line = Vec::new();
    let mut line_number: u64 = 0;
    loop {
        let read = read_line(&mut input, &mut line, max_request_bytes)?;
        line_number += 1;
        let answer = match read {
            Line::End => return Ok(()),
            Line::TooLong => protocol::error_answer(
                None,
                "request_too_large",
                &format!("the request is longer than {max_request_bytes} bytes"),
            ),
            Line::Read if is_blank(&line) => continue,
            Line::Read => protocol::answer(predictor, model_id, max_threads, &line),
        };
        match write_answer(&mut output, &answer.json) {
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
            written => written?,
        }
        if let Some(error) = &answer.error {
            // As in `run`, the log does not stop the answers.
            let _ = writeln!(log, "shindo-est: line {line_number}: {error}");
        }
    }
}

/// What [`read_line`] read.
enum Line {
    /// A line, now in the buffer without its line break.
    Read,
    /// A line longer than the limit, which was skipped.
    TooLong,
    /// The end of the input.
    End,
}

/// Reads the next line of `input` into `line`, without the line break, unless
/// it is longer than `max_bytes`.
fn read_line(input: &mut impl BufRead, line: &mut Vec<u8>, max_bytes: usize) -> io::Result<Line> {
    line.clear();
    let mut started = false;
    let mut too_long = false;
    loop {
        let available = match input.fill_buf() {
            Ok(available) => available,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if available.is_empty() {
            return Ok(match (started, too_long) {
                (false, _) => Line::End,
                (true, false) => Line::Read,
                (true, true) => Line::TooLong,
            });
        }
        started = true;
        let line_break = available.iter().position(|&byte| byte == b'\n');
        let content = &available[..line_break.unwrap_or(available.len())];
        if !too_long {
            if line.len() + content.len() > max_bytes {
                too_long = true;
                line.clear();
            } else {
                line.extend_from_slice(content);
            }
        }
        let consumed = line_break.map_or(available.len(), |index| index + 1);
        input.consume(consumed);
        if line_break.is_some() {
            return Ok(if too_long { Line::TooLong } else { Line::Read });
        }
    }
}

/// Returns whether `line` has only JSON whitespace.
fn is_blank(line: &[u8]) -> bool {
    line.iter()
        .all(|byte| matches!(byte, b' ' | b'\t' | b'\r' | b'\n'))
}

fn write_answer(output: &mut impl Write, json: &str) -> io::Result<()> {
    output.write_all(json.as_bytes())?;
    output.write_all(b"\n")?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};

    use crate::test_support;

    /// A request with `id`, short enough for a limit of 100 bytes.
    fn request(id: u32) -> String {
        format!(
            r#"{{"id":{id},"hypocenter":{{"latitude":35,"longitude":140,"depth":80,"magnitude":6.5}},"sites":[]}}"#
        )
    }

    struct Served {
        answers: Vec<String>,
        log: String,
    }

    fn serve(input: impl io::BufRead, max_request_bytes: usize) -> Served {
        let bundle = test_support::bundle();
        let mut output = Vec::new();
        let mut log = Vec::new();
        let served = super::serve(
            &bundle,
            None,
            input,
            &mut output,
            &mut log,
            max_request_bytes,
        );
        served.expect("serving from memory does not fail");
        let output = String::from_utf8(output).expect("the answers are UTF-8");
        assert!(output.is_empty() || output.ends_with('\n'), "{output:?}");
        Served {
            answers: output.lines().map(str::to_owned).collect(),
            log: String::from_utf8(log).expect("the log is UTF-8"),
        }
    }

    #[test]
    fn answers_each_line_in_order() {
        let input = format!("{}\n{}\n{}\n", request(1), request(2), request(3));

        let served = serve(input.as_bytes(), super::MAX_REQUEST_BYTES);

        assert_eq!(served.answers.len(), 3, "{:?}", served.answers);
        let numbered = served.answers.iter().zip(1..);
        for (answer, id) in numbered {
            assert!(answer.starts_with(&format!(r#"{{"id":{id},"#)), "{answer}");
        }
        assert_eq!(served.log, "");
    }

    #[test]
    fn skips_blank_lines() {
        let input = format!("\n{}\n   \n\r\n\t\n{}\n\n", request(1), request(2));

        let served = serve(input.as_bytes(), super::MAX_REQUEST_BYTES);

        assert_eq!(served.answers.len(), 2, "{:?}", served.answers);
        assert_eq!(served.log, "");
    }

    #[test]
    fn reads_crlf_lines() {
        let lf = format!("{}\n{}\n", request(1), request(2));
        let crlf = format!("{}\r\n{}\r\n", request(1), request(2));

        let served = serve(crlf.as_bytes(), super::MAX_REQUEST_BYTES);

        let expected = serve(lf.as_bytes(), super::MAX_REQUEST_BYTES);
        assert_eq!(served.answers, expected.answers);
        assert_eq!(served.log, "");
    }

    #[test]
    fn a_bad_line_does_not_stop_the_rest() {
        let input = format!("{}\ngarbage\n{}\n", request(1), request(3));

        let served = serve(input.as_bytes(), super::MAX_REQUEST_BYTES);

        assert_eq!(served.answers.len(), 3, "{:?}", served.answers);
        assert!(
            served.answers[0].starts_with(r#"{"id":1,"#),
            "{:?}",
            served.answers
        );
        assert!(
            served.answers[1].starts_with(r#"{"error":{"code":"invalid_json","#),
            "{:?}",
            served.answers
        );
        assert!(
            served.answers[2].starts_with(r#"{"id":3,"#),
            "{:?}",
            served.answers
        );
        assert!(
            served.log.starts_with("shindo-est: line 2: invalid_json: "),
            "{}",
            served.log
        );
        assert_eq!(served.log.lines().count(), 1, "{}", served.log);
    }

    #[test]
    fn rejects_a_line_over_the_limit_and_goes_on() {
        let limit = 100;
        let short = request(2);
        assert!(short.len() <= limit, "{short}");
        let input = format!(
            "{}\n{short}\n{}\n",
            "x".repeat(limit + 1),
            "x".repeat(limit)
        );

        let served = serve(input.as_bytes(), limit);

        assert_eq!(served.answers.len(), 3, "{:?}", served.answers);
        assert!(
            served.answers[0].starts_with(r#"{"error":{"code":"request_too_large","#),
            "{:?}",
            served.answers
        );
        assert!(
            served.answers[1].starts_with(r#"{"id":2,"#),
            "{:?}",
            served.answers
        );
        // A line of exactly the limit is read.
        assert!(
            served.answers[2].starts_with(r#"{"error":{"code":"invalid_json","#),
            "{:?}",
            served.answers
        );
        assert!(
            served
                .log
                .starts_with("shindo-est: line 1: request_too_large: "),
            "{}",
            served.log
        );
        assert!(
            served.log.contains("\nshindo-est: line 3: invalid_json: "),
            "{}",
            served.log
        );
    }

    #[test]
    fn reads_lines_in_small_chunks() {
        let limit = 100;
        let input = format!(
            "{}\n{}\n{}\n\n{}",
            "x".repeat(limit + 1),
            request(2),
            "x".repeat(limit),
            request(4)
        );
        let whole = serve(input.as_bytes(), limit);

        // Lines that span several reads of the buffer, and a line that goes
        // over the limit in a later read than its first.
        let chunked = serve(io::BufReader::with_capacity(16, input.as_bytes()), limit);

        assert_eq!(chunked.answers, whole.answers);
        assert_eq!(chunked.log, whole.log);
        assert_eq!(whole.answers.len(), 4, "{:?}", whole.answers);
        assert!(
            whole.answers[0].starts_with(r#"{"error":{"code":"request_too_large","#),
            "{:?}",
            whole.answers
        );
        assert!(
            whole.answers[3].starts_with(r#"{"id":4,"#),
            "{:?}",
            whole.answers
        );
    }

    #[test]
    fn a_line_break_in_a_message_stays_in_one_log_line() {
        let input = format!("{}\n{}\n", r#"{"id": 1, "foo\nbar": 1}"#, request(2));

        let served = serve(input.as_bytes(), super::MAX_REQUEST_BYTES);

        assert_eq!(served.answers.len(), 2, "{:?}", served.answers);
        assert!(
            served.answers[1].starts_with(r#"{"id":2,"#),
            "{:?}",
            served.answers
        );
        assert_eq!(served.log.lines().count(), 1, "{}", served.log);
        assert!(
            served
                .log
                .starts_with("shindo-est: line 1: invalid_request: "),
            "{}",
            served.log
        );
    }

    #[test]
    fn answers_a_last_line_without_a_newline() {
        let input = format!("{}\n{}", request(1), request(2));

        let served = serve(input.as_bytes(), super::MAX_REQUEST_BYTES);

        assert_eq!(served.answers.len(), 2, "{:?}", served.answers);
        assert!(
            served.answers[1].starts_with(r#"{"id":2,"#),
            "{:?}",
            served.answers
        );
    }

    /// An output that records, at each flush, how many lines it has.
    #[derive(Default)]
    struct FlushRecorder {
        bytes: Vec<u8>,
        lines_at_flush: Vec<usize>,
    }

    impl Write for FlushRecorder {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            let lines = String::from_utf8_lossy(&self.bytes).matches('\n').count();
            self.lines_at_flush.push(lines);
            Ok(())
        }
    }

    #[test]
    fn flushes_after_each_answer() {
        let input = format!("{}\ngarbage\n{}\n", request(1), request(3));
        let bundle = test_support::bundle();
        let mut output = FlushRecorder::default();

        let served = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            &mut output,
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );

        served.expect("serving from memory does not fail");
        assert_eq!(output.lines_at_flush, [1, 2, 3]);
    }

    /// An output whose writes fail with `kind`.
    struct FailingOutput(io::ErrorKind);

    impl Write for FailingOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(self.0.into())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(self.0.into())
        }
    }

    #[test]
    fn a_closed_output_ends_quietly() {
        let input = format!("{}\n{}\n", request(1), request(2));
        let bundle = test_support::bundle();

        let closed = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            FailingOutput(io::ErrorKind::BrokenPipe),
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );
        let failed = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            FailingOutput(io::ErrorKind::PermissionDenied),
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );

        assert!(closed.is_ok(), "{closed:?}");
        let error = failed.expect_err("a failed write is an error");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }

    /// An output that takes every write but whose flushes fail with `kind`.
    struct FailingFlush(io::ErrorKind);

    impl Write for FailingFlush {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(self.0.into())
        }
    }

    #[test]
    fn a_failed_flush_is_a_failed_write() {
        let input = format!("{}\n{}\n", request(1), request(2));
        let bundle = test_support::bundle();

        let closed = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            FailingFlush(io::ErrorKind::BrokenPipe),
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );
        let failed = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            FailingFlush(io::ErrorKind::PermissionDenied),
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );

        assert!(closed.is_ok(), "{closed:?}");
        let error = failed.expect_err("a failed flush is an error");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn a_failed_log_does_not_stop_the_answers() {
        let input = format!("{}\ngarbage\n{}\n", request(1), request(3));
        let bundle = test_support::bundle();
        let mut output = Vec::new();

        let served = super::serve(
            &bundle,
            None,
            input.as_bytes(),
            &mut output,
            FailingOutput(io::ErrorKind::PermissionDenied),
            super::MAX_REQUEST_BYTES,
        );

        assert!(served.is_ok(), "{served:?}");
        let answers = String::from_utf8(output).expect("the answers are UTF-8");
        assert_eq!(answers.lines().count(), 3, "{answers}");
    }

    struct FailingInput;

    impl io::Read for FailingInput {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::ErrorKind::PermissionDenied.into())
        }
    }

    #[test]
    fn a_failed_read_is_an_error() {
        let bundle = test_support::bundle();

        let served = super::serve(
            &bundle,
            None,
            io::BufReader::new(FailingInput),
            io::sink(),
            io::sink(),
            super::MAX_REQUEST_BYTES,
        );

        let error = served.expect_err("a failed read is an error");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn max_request_bytes_is_16_mib() {
        assert_eq!(super::MAX_REQUEST_BYTES, 16 * 1024 * 1024);
    }
}
