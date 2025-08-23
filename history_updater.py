- name: Run conductor
  run: python conductor.py

# Commit & push regardless of success/failure above
- name: Commit & push scraped CSVs
  if: always()
  run: |
    git config user.name "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    git add -f historical_etf_yields_*.csv || true
    git commit -m "conductor: checkpoint scraped history [${{ job.status }}]" || echo "Nothing to commit"
    git push || true
