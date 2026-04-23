## Submodules

This project imports older projects with overlapping functionality as submodules for references.

### First time

```bash
git submodule --init
git submodule --update
```

Fetch changes from upstream projects:

```bash
git submodule foreach git pull
```
