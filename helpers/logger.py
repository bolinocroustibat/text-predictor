from datetime import datetime
import typer

from typing import Optional


class CustomLogger:

    def log(self, message: str, color: Optional[str] = None) -> None:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if not color:
            color: str = "white"
        typer.secho(f"\n{dt_string} - {message}", fg=color)

    def success(self, message: str) -> None:
        self.log(message, "green")

    def error(self, message: str) -> None:
        self.log(message, "red")

    def warning(self, message: str) -> None:
        self.log(message, "yellow")

    def debug(self, message: str) -> None:
        self.log(message, "magenta")
        
    def info(self, message: str) -> None:
        self.log(message, "cyan")
