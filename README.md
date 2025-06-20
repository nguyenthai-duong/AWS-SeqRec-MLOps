# datn-recsys

**Product-ready Recommender System pipeline.**

---

## 🚀 Quickstart

```bash
# 1. Clone repo
git clone https://github.com/nguyenthai-duong/AWS-SeqRec-MLOps
cd AWS-SeqRec-MLOps

conda create -n recsys_ops python=3.11 -y
conda activate recsys_ops
pip install uv==0.6.2
uv sync --all-groups
python -m ipykernel install --user --name=datn-recsys --display-name="Python (datn-recsys)"

# 4. Cài pre-commit (khuyến nghị, auto lint trước khi commit)
make precommit

# 6. Check style/lint toàn bộ code & notebook
make style

# 7. Chạy unit test
make test


Setup pr agent github (https://github.com/qodo-ai/pr-agent)

Vào Settings → Secrets and variables → Actions -> New repository secret
Name: OPENAI_API_KEY
Value: dán OpenAI API Key của bạn

Tạo label:  Issues → Labels
Bấm New label

    Name: pr-agent/review

    Color: để mặc định hoặc chọn tuỳ ý

    Description: “Trigger PR Agent review on this PR”

Bấm Create label
Gắn label: pr-agent/review khi tạo Pull Request

![systempipline](images/setup/pr_agent.png)
