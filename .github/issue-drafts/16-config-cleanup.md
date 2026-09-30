title: "Cleanup: duplicate ENVIRONMENT singleton in src/config.py; unpinned requirements"
labels: cleanup, good-first-issue
---
- [ ] `src/config.py` assigns `ENVIRONMENT = EnvironmentConfig()` twice (end of file). Harmless but confusing; remove the second.
- [ ] `requirements.txt` uses `>=` everywhere. `constraints.txt` now pins the tested versions for Docker/CI, but local `pip install -r requirements.txt` still floats. Mention `-c constraints.txt` in README's setup instructions.
- [ ] README: add a "Deployment" section linking `docs/DEPLOYMENT.md`, and add CI/deploy to the traceability table where relevant (NFR-04 restart policy, NFR-10 secrets handling).
