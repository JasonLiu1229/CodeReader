import typer

app = typer.Typer(add_completion=False)

@app.command()
def grade():
    ...

if __name__ == "__main__":
    app()
