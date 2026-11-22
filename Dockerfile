# Shopping Sessions: native C++ pipeline built with clang, statically linked, run from an empty image.
FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang-18 g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*
COPY src/ /src/
RUN clang++-18 -O3 -march=x86-64-v3 -mtune=cascadelake -std=c++17 -pthread -static \
        -o /pipeline /src/pipeline.cpp

FROM scratch
COPY --from=build /pipeline /pipeline
USER 1000:1000
CMD ["/pipeline"]
