"""Publish the trained model and the web page to Hugging Face.

Run from the project folder, after logging in once with `hf auth login` and exporting the model
with `python -m deploy.export_onnx`:
    python deploy/deploy.py

It creates (or updates) two public repos on your Hugging Face account:
  - a model repo holding the PyTorch model, its ONNX version and the model card
  - a static Space serving web/, which downloads model.onnx from the model repo and runs it in
    the visitor's browser (static Spaces are free; Python ones need a paid plan)
Running it again uploads the current files as a new commit.
"""
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

MODEL_REPO = "bebird/cats-vs-dogs-efficientnet-b0"
SPACE_REPO = "bebird/cat-or-dog"

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"
WEB = ROOT / "web"

MODEL_FILES = {  # {path in the repo: local file}
    "best_model.pt": ROOT / "models" / "best_model.pt",
    "model.onnx": ROOT / "models" / "model.onnx",
    "README.md": DEPLOY / "model_card.md",
}
SPACE_FILES = {
    "index.html": WEB / "index.html",
    "style.css": WEB / "style.css",
    "app.js": WEB / "app.js",
    "README.md": DEPLOY / "space_README.md",
}


def publish(api, repo_id, repo_type, files, message, **create_options):
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing files, nothing uploaded: {missing}")
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, **create_options)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()]
    api.create_commit(repo_id, repo_type=repo_type, operations=operations, commit_message=message)
    prefix = "spaces/" if repo_type == "space" else ""
    print(f"Published https://huggingface.co/{prefix}{repo_id}")


def main():
    api = HfApi()
    print("Logged in as", api.whoami()["name"])
    # The model goes first, because the web page downloads it
    publish(api, MODEL_REPO, "model", MODEL_FILES, "Add ONNX model for in-browser use")
    publish(api, SPACE_REPO, "space", SPACE_FILES, "Deploy web page", space_sdk="static")
    print("Done. The Space is live within a minute.")


if __name__ == "__main__":
    main()
