# Build context is the repository root. The layout inside the image mirrors the
# repository so that `artifacts/` resolves the same way it does on the host:
# predictor.py walks three parents up from app/model/ to find it.
FROM python:3.11-slim

# weasyprint (Phase 12's PDF renderer) is a Python package over the pango/cairo
# stack, so the system libraries have to be here too — the pip install alone
# imports fine and fails at render time with "cannot load library libgobject".
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
       libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libfontconfig1 \
       libcairo2 libgdk-pixbuf-2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ml-service/requirements.txt ml-service/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r ml-service/requirements.txt

COPY ml-service/ ml-service/

# PYTHONHASHSEED is pinned because reproducibility is a project rule: any
# iteration over a set of strings that touches an RNG would otherwise give a
# different answer in every process. The known instance is fixed at the source
# (closed_loop.py sorts its concept set); this is the standing guard.
ENV PYTHONPATH=/workspace/ml-service \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
