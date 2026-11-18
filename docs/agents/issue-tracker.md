# Issue tracker: GitHub

Use GitHub Issues in `TheSlowpes/genkit-cowork`. Run `gh issue` commands with
`--repo TheSlowpes/genkit-cowork`.

- Publish an issue or spec: `gh issue create --title "..." --body-file -`.
- Read an issue: `gh issue view <number> --json number,title,body,labels,author,createdAt,updatedAt`.
- Comment, label, or close: use `gh issue comment`, `gh issue edit`, or `gh issue close`.

For a complete oldest-first open issue queue, paginate the REST API and exclude PRs:

```sh
gh api --paginate 'repos/TheSlowpes/genkit-cowork/issues?state=open&sort=created&direction=asc&per_page=100' \
  --jq '.[] | select(.pull_request == null)'
```

Add `labels=<URL-encoded label>` to filter the queue. Before evaluating reporter
activity, fetch the full discussion:

```sh
gh api --paginate 'repos/TheSlowpes/genkit-cowork/issues/<number>/comments?per_page=100'
```

Report retrieval failures rather than treating the queue or discussion as complete.

**PRs as a request surface: no.**
