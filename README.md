# Fact survival forecasting (Neural Landmark Hazard)


Rolling origin: the model is refitted from scratch at every origin on the labels known by then, and predicts the facts active
at that origin. `--part i --parts n` splits the origins over several jobs.

## Tests
```bash
FINWIKI_PKL=data/FinWiki/train_valid_test.pkl python -m pytest tests
```
Tests that need a data file are skipped when its environment variable is not set.

## Known open points
pip install -r requirements.txt
python main.py develop configs/finwiki.json --data data/FinWiki/train_valid_test.pkl --out results/finwiki --device cuda
python main.py select  configs/finwiki.json --out results/finwiki
python main.py test    configs/finwiki.json --data data/FinWiki/train_valid_test.pkl --out results/finwiki --device cuda
python main.py report  configs/finwiki.json --out results/finwiki
