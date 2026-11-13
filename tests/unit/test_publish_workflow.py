from pathlib import Path


def test_publish_workflow_commits_tag_derived_version_before_publish() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github/workflows/publish.yml").read_text()

    assert "types: [published]" in workflow
    assert "if: github.repository == 'dynotx/dynopsi'" in workflow
    assert "contents: write" in workflow
    assert "id-token: write" in workflow
    assert "name: pypi" in workflow
    assert "ref: main" in workflow
    assert "token: ${{ secrets.PUBLISH_PAT }}" in workflow
    assert "RELEASE_TAG: ${{ github.event.release.tag_name }}" in workflow
    assert "lowercase-v stable semantic version" in workflow
    assert 'echo "RELEASE_VERSION=${RELEASE_TAG#v}" >> "$GITHUB_ENV"' in workflow
    assert 'uv version "$RELEASE_VERSION"' in workflow
    assert "uv lock --check" in workflow
    assert "git add pyproject.toml uv.lock" in workflow
    assert "git diff --cached --quiet" in workflow
    assert "git push origin HEAD:main" in workflow
    assert "git add ." not in workflow

    build = workflow.index("run: uv build")
    commit = workflow.index("git commit")
    push = workflow.index("git push origin HEAD:main")
    publish = workflow.index("uses: pypa/gh-action-pypi-publish@release/v1")
    assert build < commit < push < publish
