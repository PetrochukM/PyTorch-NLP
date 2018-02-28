if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
  TRAVIS_COMMIT_RANGE="FETCH_HEAD...$TRAVIS_BRANCH"
fi
git diff --name-only $TRAVIS_COMMIT_RANGE | grep -qvE '(\.md$)|(^(docs|examples))/' || {
  echo "Only docs were updated, stopping build process."
  exit
}