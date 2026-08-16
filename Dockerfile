# Slim base image: full python:3.11 includes compilers/docs we don't need
# at runtime; slim keeps the deployed image smaller and faster to pull on
# Render's free/starter tiers. Pinned to 3.11 (not 3.14) deliberately -
# it's the version scikit-learn/xgboost/pandas have stable, prebuilt
# wheels for, so the build doesn't fall back to compiling from source.
FROM python:3.11-slim

WORKDIR /app

# Copy requirements BEFORE the rest of the source. Docker caches layers,
# so as long as requirements.txt hasn't changed, `docker build` reuses
# the cached pip install layer instead of re-downloading every package
# every time you change one line of app code - this is the single
# biggest thing that makes Python Docker builds fast to iterate on.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and the pre-trained model artifact.
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Render sets $PORT at runtime and expects the service to bind to it -
# it is NOT always 8000, so don't hardcode the port. The shell form of
# CMD is required here (not the exec-array form) because $PORT only gets
# substituted when the command runs through a shell.
ENV PORT=8000
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
