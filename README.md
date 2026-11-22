# Shopping Sessions — native pipeline

A single C++17 program (`src/pipeline.cpp`), statically linked into an empty image.

Phases (all multi-threaded, `NCPU` threads):
1. **Links** — parse `links/*.csv`, hash every ID into a lock-free open-addressing table, lock-free union-find, then each component's byte-wise smallest ID is its name.
2. **Events + clicks** — mmap'd CSV scanned with SIMD separator/quote masks; every well-formed event line becomes a 16-byte dedup record (hashed `event_id`, `ingest_ts`) scattered by event-id hash, and every valid line a 24-byte person record scattered by person hash. Valid clicks are scattered by the same person hash.
3. **Dedup** — per event-id partition, the copy with the smallest `ingest_ts` wins; losing copies are marked dead in place.
4. **Persons** — per person partition: group, sort by `ts`, sessionize, credit the latest click within 7 days (smallest `click_id` on ties), and count assists from merged per-campaign 7-day intervals.
5. **Output** — a minimal hand-written Parquet writer (PLAIN, uncompressed), one `user_daily` file per thread and one `campaign_daily` file.

Set `PIPE_VERBOSE=1` for phase timings on stderr.
