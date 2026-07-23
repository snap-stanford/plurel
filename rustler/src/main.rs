use clap::{Parser, Subcommand};

mod common;
pub mod fly;
#[cfg(feature = "pre")]
mod pre;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[cfg(feature = "pre")]
    Pre(pre::Cli),
    Fly(fly::Cli),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "pre")]
        Command::Pre(cli) => {
            pre::main(cli);
        }
        Command::Fly(cli) => {
            fly::main(cli);
        }
    }
}
