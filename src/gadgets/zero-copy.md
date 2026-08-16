---
title: One D2D copy away from zero-copy
date: 2026-08-12
blurb: NvSci camera pipeline holding at ~150µs.
---

The NvSci zero-copy path on the sensor HIL rig is working. End-to-end it sits
around 150µs, which is where I wanted it. There's still one device-to-device
copy in the middle that shouldn't need to exist.

## Where the copy comes from

Short version: the producer and consumer negotiate buffer attributes separately,
and when their alignment requirements don't match exactly, the driver inserts a
copy rather than failing. It's silent. Nothing in the logs says "I copied your
buffer" — you only see it as time.

## What's next

Getting the attribute lists to agree at allocation time, so the same memory is
valid for both sides. That's the whole remaining gap.
