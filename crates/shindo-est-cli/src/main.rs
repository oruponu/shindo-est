//! The `shindo-est` command, which packs and inspects model files and answers
//! prediction requests with them.
//!
//! It exits with 0 on success, which for `predict` is the end of its input, 1
//! when a command fails (with the error and its causes on standard error), and
//! 2 when the command line does not follow the usage (with the usage on
//! standard error).

mod args;
mod info;
mod pack;
mod protocol;
mod report;
mod serve;
#[cfg(test)]
mod test_support;

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

use crate::args::Command;

/// Exit code of a command line that does not follow the usage.
const USAGE_EXIT_CODE: u8 = 2;

fn main() -> ExitCode {
    let command = match args::parse(env::args_os().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            report_error(&format!("{error}\n{}", args::USAGE));
            return ExitCode::from(USAGE_EXIT_CODE);
        }
    };
    let output = match command {
        Command::Help => Ok(format!("{}\n", args::USAGE)),
        Command::Version => Ok(format!("shindo-est {}\n", env!("CARGO_PKG_VERSION"))),
        Command::Info { model } => info::run(&model),
        Command::Pack(paths) => pack::run(&paths),
        // The answers are already written.
        Command::Predict { model, max_threads } => {
            serve::run(&model, max_threads).map(|()| String::new())
        }
    };
    let written = output.and_then(|text| {
        io::stdout().write_all(text.as_bytes())?;
        Ok(())
    });
    match written {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            report_error(&report::describe(error.as_ref()));
            ExitCode::FAILURE
        }
    }
}

/// Writes "shindo-est: <message>" to standard error. Unlike `eprintln!`, it
/// does not panic when standard error is closed, so that the exit code stays
/// the one that tells what went wrong.
fn report_error(message: &str) {
    // There is nowhere left to report a failure to write the report.
    let _ = writeln!(io::stderr(), "shindo-est: {message}");
}
