# GLM-5.3-Flash on llama.cpp for sm120: Unsloth's glm5next branch at the commit our
# measurements used, plus the five patches in patches/ (applied with git am).
# Upstream thread: ggml-org/llama.cpp#27754 (Unsloth's branch glm5next/upstream).
# Retirement condition: when #27754 merges WITH the can_reuse and ssm_a fixes, drop the
# clone override and the patches. Mirrors the fork's .devops/cuda.Dockerfile server target,
# pinned base digests, sm120 only.
ARG LLAMA_REPO=https://github.com/unslothai/llama.cpp
ARG LLAMA_SHA=2e0e57f10
ARG CUDA_DOCKER_ARCH=120

FROM docker.io/node:24@sha256:be23f54a88d34e8824c741b19b91064094f92c1c97b194144bfc8b50d67258e2 AS web
ARG LLAMA_REPO
ARG LLAMA_SHA
RUN git clone ${LLAMA_REPO} /src && git -C /src checkout ${LLAMA_SHA}
WORKDIR /src/tools/ui
RUN npm ci && LLAMA_BUILD_NUMBER=${LLAMA_SHA} npm run build

FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS build
ARG LLAMA_REPO
ARG LLAMA_SHA
ARG CUDA_DOCKER_ARCH
RUN apt-get update && apt-get install -y gcc-14 g++-14 build-essential cmake python3 python3-pip git libssl-dev libgomp1
ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
RUN git clone ${LLAMA_REPO} /app && git -C /app checkout ${LLAMA_SHA}
WORKDIR /app
COPY patches/*.patch /patches/
RUN git config user.email build@local && git config user.name build && git am /patches/*.patch
COPY --from=web /src/tools/ui/dist tools/ui/dist
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . \
    && cmake --build build --config Release -j$(nproc)
RUN mkdir -p /app/lib && find build -name "*.so*" -exec cp -P {} /app/lib \; \
    && mkdir -p /app/full && cp build/bin/* /app/full

FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867 AS server
RUN apt-get update && apt-get install -y libgomp1 curl && apt clean -y && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/lib/ /app
COPY --from=build /app/full/llama /app/full/llama-server /app
WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8000/health" ]
ENTRYPOINT [ "/app/llama-server" ]
