//! Runs `shindo-est predict` on the model file fixture.

use std::env;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// How long a test waits for an answer before it fails, as a process that
/// does not flush its answers would otherwise hang the test.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(10);

/// A request for the fixture's hypocenter at one site.
const REQUEST: &str = r#"{"id": 1, "hypocenter": {"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}, "sites": [{"latitude": 35.02, "longitude": 140.02, "arv": 1.0}]}"#;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bundle")
        .join(name)
}

fn command(arguments: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_shindo-est"));
    command.args(arguments);
    command
}

fn start(arguments: &[&str]) -> Child {
    let model = fixture("model.bin");
    let mut command = command(&["predict", "--model"]);
    command
        .arg(model)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.spawn().expect("the binary should start")
}

/// Runs `predict` on the fixture with `input` and `arguments` after
/// `--model`, until the input ends.
fn run_with_input(input: &str, arguments: &[&str]) -> Output {
    let mut child = start(arguments);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let input = input.to_owned();
    // The input is written on its own thread while this one reads the
    // answers, as the process stops reading when its full output pipe is
    // not read.
    let writer = thread::spawn(move || {
        stdin
            .write_all(input.as_bytes())
            .expect("the requests should be writable");
    });
    let output = child.wait_with_output().expect("the binary should finish");
    writer.join().expect("the requests are written");
    output
}

/// Sends the lines of `stdout` over a channel, so that a test can wait for
/// each with a timeout.
fn lines(stdout: ChildStdout) -> mpsc::Receiver<String> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    receiver
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("the errors are UTF-8")
}

#[test]
fn stays_resident_between_requests() {
    let mut child = start(&[]);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let answers = lines(child.stdout.take().expect("stdout is piped"));

    for id in [1, 2] {
        let request = REQUEST.replacen(r#""id": 1"#, &format!(r#""id": {id}"#), 1);
        writeln!(stdin, "{request}").expect("the request should be writable");
        stdin.flush().expect("the request should be sent");

        let answer = answers.recv_timeout(ANSWER_TIMEOUT);

        let answer = answer.unwrap_or_else(|error| panic!("no answer to request {id}: {error}"));
        assert!(answer.starts_with(&format!(r#"{{"id":{id},"#)), "{answer}");
    }
    drop(stdin);
    let output = child.wait_with_output().expect("the binary should finish");
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(
        errors.contains("shindo-est: loaded model synthetic-fixture from "),
        "{errors}"
    );
}

#[test]
fn max_threads_does_not_change_the_answers() {
    // Enough sites for several threads.
    let mut sites = String::new();
    for index in 0..1_000_u16 {
        if index > 0 {
            sites.push_str(", ");
        }
        let latitude = 34.8 + f64::from(index % 40) * 0.01;
        let longitude = 140.0 + f64::from(index / 40) * 0.01;
        write!(
            sites,
            r#"{{"latitude": {latitude}, "longitude": {longitude}, "arv": 1.0}}"#
        )
        .expect("writing to a string does not fail");
    }
    let request = format!(
        r#"{{"hypocenter": {{"latitude": 35.02, "longitude": 140.02, "depth": 80.0, "magnitude": 6.5}}, "sites": [{sites}]}}"#
    );
    let input = format!("{request}\n{request}\n");

    let one = run_with_input(&input, &["--max-threads", "1"]);
    let automatic = run_with_input(&input, &[]);

    assert!(one.status.success(), "{}", stderr(&one));
    assert!(automatic.status.success(), "{}", stderr(&automatic));
    let answers = String::from_utf8(one.stdout).expect("the answers are UTF-8");
    assert_eq!(answers.lines().count(), 2, "{answers}");
    assert!(answers.as_bytes() == automatic.stdout, "the answers differ");
}

#[test]
fn a_missing_model_exits_with_1() {
    let missing = fixture("missing.bin");
    let missing_text = missing.display().to_string();

    let output = command(&["predict", "--model", &missing_text])
        .stdin(Stdio::null())
        .output()
        .expect("the binary should run");

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let errors = stderr(&output);
    assert!(errors.contains(&missing_text), "{errors}");
}

#[test]
fn bad_predict_arguments_exit_with_2() {
    for arguments in [
        &["predict"][..],
        &["predict", "--model"],
        &["predict", "--model", "model.bin", "--max-threads", "0"],
    ] {
        let output = command(arguments)
            .stdin(Stdio::null())
            .output()
            .expect("the binary should run");

        assert_eq!(
            output.status.code(),
            Some(2),
            "{arguments:?}: {}",
            stderr(&output)
        );
    }
}
