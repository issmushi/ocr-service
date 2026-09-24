import assert from "node:assert/strict";
import { File as NodeFile } from "node:buffer";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import ts from "typescript";

// Use the project's existing compiler so these tests also run on Node 18/20.
const source = await readFile(new URL("../src/components/cameraBurst.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
});
const { captureCameraFrames, waitForCameraFrame } = await import(
  `data:text/javascript;base64,${Buffer.from(`${outputText}\n//# sourceURL=cameraBurst.test-source.js\n`).toString("base64")}`
);
globalThis.File ??= NodeFile;

function mockVideo({ nativeFrames = true, ready = true } = {}) {
  const callbacks = new Map();
  let frameId = 0;
  const video = {
    readyState: ready ? 2 : 1,
    videoWidth: ready ? 1920 : 0,
    videoHeight: ready ? 1080 : 0,
    currentTime: 0,
    callbacks,
    cancelled: [],
    requestVideoFrameCallback(callback) {
      callbacks.set(++frameId, callback);
      return frameId;
    },
    cancelVideoFrameCallback(id) {
      video.cancelled.push(id);
      callbacks.delete(id);
    },
    present(time) {
      video.currentTime = time;
      const pending = [...callbacks.values()];
      callbacks.clear();
      for (const callback of pending) callback(performance.now(), { mediaTime: time });
    },
  };
  if (!nativeFrames) delete video.requestVideoFrameCallback;
  return video;
}

function streamingVideo(t, options) {
  const video = mockVideo(options);
  const started = performance.now();
  const timer = setInterval(() => video.present((performance.now() - started) / 1000), 10);
  t.after(() => clearInterval(timer));
  return video;
}

function mockCanvas(t, {
  failAt,
  blobFailure = "null",
  fallbackFailure,
  fallbackFailAt,
  holdEncoding = false,
  mimeType = "image/jpeg",
  fallbackBytes,
} = {}) {
  const previousDocument = globalThis.document;
  const draws = [];
  const encodings = [];
  const fallbacks = [];
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => ({
      drawImage(video) {
        draws.push({ at: performance.now(), mediaTime: video.currentTime, width: canvas.width, height: canvas.height });
      },
    }),
    toBlob(callback, type, quality) {
      assert.equal(type, "image/jpeg");
      assert.equal(quality, 0.94);
      const photo = new Blob([`frame-${draws.at(-1).mediaTime}`], { type: mimeType });
      const shouldFail = draws.length === failAt;
      encodings.push(() => callback(shouldFail ? blobFailure === "empty" ? new Blob([], { type }) : null : photo));
      if (shouldFail && blobFailure === "throw") throw new Error("Native async encoding failed.");
      if (shouldFail && blobFailure === "security") throw new DOMException("Canvas is not origin-clean.", "SecurityError");
      if (!holdEncoding) encodings.at(-1)();
    },
    toDataURL(type, quality) {
      assert.equal(type, "image/jpeg");
      assert.equal(quality, 0.94);
      fallbacks.push({ mediaTime: draws.at(-1).mediaTime, width: canvas.width, height: canvas.height });
      if (fallbackFailure && (fallbackFailAt === undefined || draws.length === fallbackFailAt)) {
        if (fallbackFailure === "throw") throw new Error("Native sync encoding failed.");
        if (fallbackFailure === "security") throw new DOMException("Canvas is not origin-clean.", "SecurityError");
        return fallbackFailure;
      }
      return `data:${mimeType};base64,${Buffer.from(fallbackBytes ?? `frame-${draws.at(-1).mediaTime}`).toString("base64")}`;
    },
  };
  if (blobFailure === "missing") delete canvas.toBlob;
  globalThis.document = { createElement: (tag) => { assert.equal(tag, "canvas"); return canvas; } };
  t.after(() => {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  });
  return { canvas, draws, encodings, fallbacks };
}

function trackEncodingTimeouts(t) {
  const original = globalThis.setTimeout;
  const callbacks = [];
  globalThis.setTimeout = (callback, milliseconds, ...args) => {
    if (milliseconds === 5_000) callbacks.push(() => callback(...args));
    return original(callback, milliseconds, ...args);
  };
  t.after(() => { globalThis.setTimeout = original; });
  return callbacks;
}

test("native frame readiness ignores metadata and repeated timestamps until a real new frame", async () => {
  const video = mockVideo({ ready: false });
  const controller = new AbortController();
  let resolved = false;
  const frame = waitForCameraFrame(video, controller.signal, { previousMediaTime: 1 }).then((time) => {
    resolved = true;
    return time;
  });
  video.present(2);
  await Promise.resolve();
  assert.equal(resolved, false);
  video.readyState = 2;
  video.videoWidth = 1920;
  video.videoHeight = 1080;
  video.present(1);
  await Promise.resolve();
  assert.equal(resolved, false);
  video.present(3);
  assert.equal(await frame, 3);
  assert.equal(video.callbacks.size, 0);
});

test("fallback waits for readiness and advancing video time", async () => {
  const video = mockVideo({ nativeFrames: false, ready: false });
  const controller = new AbortController();
  let resolved = false;
  const frame = waitForCameraFrame(video, controller.signal, { previousMediaTime: 1 }).then((time) => {
    resolved = true;
    return time;
  });
  video.currentTime = 2;
  await delay(40);
  assert.equal(resolved, false);
  video.readyState = 2;
  video.videoWidth = 1280;
  video.videoHeight = 720;
  video.currentTime = 1;
  await delay(40);
  assert.equal(resolved, false);
  video.currentTime = 3;
  assert.equal(await frame, 3);
});

for (const nativeFrames of [true, false]) {
  test(`captures three distinct frames >=350ms apart (${nativeFrames ? "native callbacks" : "fallback"})`, async (t) => {
    const video = streamingVideo(t, { nativeFrames });
    const { draws } = mockCanvas(t);
    const progress = [];
    const files = await captureCameraFrames(video, 3, new AbortController().signal, (count) => progress.push(count));
    assert.equal(files.length, 3);
    assert.deepEqual(progress, [1, 2, 3]);
    assert.equal(new Set(files.map((file) => file.name)).size, 3);
    assert.equal(new Set(await Promise.all(files.map((file) => file.text()))).size, 3);
    for (let index = 1; index < draws.length; index += 1) {
      assert.ok(draws[index].at - draws[index - 1].at >= 350, JSON.stringify(draws));
      assert.ok(draws[index].mediaTime > draws[index - 1].mediaTime);
    }
    assert.equal(video.callbacks.size, 0);
  });
}

test("regular capture produces only one frame", async (t) => {
  const video = streamingVideo(t);
  const { draws } = mockCanvas(t);
  const progress = [];
  const files = await captureCameraFrames(video, 1, new AbortController().signal, (count) => progress.push(count));
  assert.equal(files.length, 1);
  assert.equal(draws.length, 1);
  assert.deepEqual(progress, [1]);
});

test("failed second encoding rejects the partial burst and a fresh attempt succeeds", async (t) => {
  const video = streamingVideo(t);
  const { draws } = mockCanvas(t, { failAt: 2, fallbackFailure: "throw", fallbackFailAt: 2 });
  const progress = [];
  await assert.rejects(
    captureCameraFrames(video, 3, new AbortController().signal, (count) => progress.push(count)),
    /Не удалось сохранить фото/,
  );
  assert.equal(draws.length, 2);
  assert.deepEqual(progress, [1]);
  const retry = await captureCameraFrames(video, 3, new AbortController().signal, () => {});
  assert.equal(retry.length, 3);
});

test("cancelling after the first photo prevents the rest of the burst", async (t) => {
  const video = streamingVideo(t);
  const { draws } = mockCanvas(t);
  const controller = new AbortController();
  const progress = [];
  await assert.rejects(captureCameraFrames(video, 3, controller.signal, (count) => {
    progress.push(count);
    controller.abort();
  }), { name: "AbortError" });
  await delay(400);
  assert.equal(draws.length, 1);
  assert.deepEqual(progress, [1]);
  assert.equal(video.callbacks.size, 0);
});

test("cancelling encoding ignores a late toBlob callback", async (t) => {
  const video = mockVideo();
  const { encodings } = mockCanvas(t, { holdEncoding: true });
  const controller = new AbortController();
  const progress = [];
  const capture = captureCameraFrames(video, 3, controller.signal, (count) => progress.push(count));
  video.present(1);
  await Promise.resolve();
  assert.equal(encodings.length, 1);
  controller.abort();
  encodings[0]();
  await assert.rejects(capture, { name: "AbortError" });
  assert.deepEqual(progress, []);
});

test("cancelling a frame wait removes its callback and ignores a late callback", async () => {
  const video = mockVideo();
  const controller = new AbortController();
  const frame = waitForCameraFrame(video, controller.signal);
  const lateCallback = [...video.callbacks.values()][0];
  controller.abort();
  lateCallback(performance.now(), { mediaTime: 1 });
  await assert.rejects(frame, { name: "AbortError" });
  assert.deepEqual(video.cancelled, [1]);
  assert.equal(video.callbacks.size, 0);
});

test("cancelling during the spacing delay never schedules another video frame", async () => {
  const video = mockVideo();
  const controller = new AbortController();
  const frame = waitForCameraFrame(video, controller.signal, { minDelayMs: 50 });
  controller.abort();
  await assert.rejects(frame, { name: "AbortError" });
  await delay(70);
  assert.equal(video.callbacks.size, 0);
});

test("a stalled camera times out and cancels its pending frame callback", async () => {
  const video = mockVideo();
  await assert.rejects(waitForCameraFrame(video, new AbortController().signal, { timeoutMs: 20 }), /новые кадры/);
  assert.equal(video.callbacks.size, 0);
  assert.deepEqual(video.cancelled, [1]);
});

for (const blobFailure of ["null", "throw", "empty", "missing"]) {
  test(`recovers from ${blobFailure} async encoding using the same full-resolution frame`, async (t) => {
    const video = mockVideo();
    const { canvas, draws, fallbacks } = mockCanvas(t, { failAt: 1, blobFailure });
    const progress = [];
    const pending = captureCameraFrames(video, 1, new AbortController().signal, (count) => progress.push(count));
    video.present(1.25);
    const files = await pending;
    assert.equal(files.length, 1);
    assert.deepEqual(progress, [1]);
    assert.equal(draws.length, 1);
    assert.equal(await files[0].text(), "frame-1.25");
    assert.deepEqual(fallbacks, [{ mediaTime: 1.25, width: 1920, height: 1080 }]);
    assert.equal(canvas.width, 0);
    assert.equal(canvas.height, 0);
  });
}

test("a transient second-frame encoding failure keeps all three distinct photos", async (t) => {
  const video = streamingVideo(t);
  const { draws, fallbacks, encodings } = mockCanvas(t, { failAt: 2 });
  const progress = [];
  const files = await captureCameraFrames(video, 3, new AbortController().signal, (count) => progress.push(count));
  assert.deepEqual(progress, [1, 2, 3]);
  assert.equal(draws.length, 3);
  assert.equal(files.length, 3);
  assert.equal(encodings.length, 2);
  assert.equal(fallbacks.length, 2);
  assert.deepEqual(await Promise.all(files.map((file) => file.text())), draws.map((draw) => `frame-${draw.mediaTime}`));
  assert.equal(new Set(draws.map((draw) => draw.mediaTime)).size, 3);
  for (const frame of draws) assert.deepEqual([frame.width, frame.height], [1920, 1080]);
});

test("a stalled toBlob callback falls back once and ignores its eventual result", async (t) => {
  const video = mockVideo();
  const { encodings, fallbacks, draws } = mockCanvas(t, { holdEncoding: true });
  const timeoutCallbacks = trackEncodingTimeouts(t);
  const progress = [];
  const pending = captureCameraFrames(video, 1, new AbortController().signal, (count) => progress.push(count));
  video.present(2.5);
  await Promise.resolve();
  assert.equal(encodings.length, 1);
  timeoutCallbacks.at(-1)();
  const files = await pending;
  encodings[0]();
  timeoutCallbacks.at(-1)();
  assert.equal(files.length, 1);
  assert.equal(await files[0].text(), "frame-2.5");
  assert.deepEqual(progress, [1]);
  assert.equal(draws.length, 1);
  assert.equal(fallbacks.length, 1);
});

test("cancellation prevents a pending timeout from starting the fallback encoder", async (t) => {
  const video = mockVideo();
  const { canvas, encodings, fallbacks } = mockCanvas(t, { holdEncoding: true });
  const timeoutCallbacks = trackEncodingTimeouts(t);
  const controller = new AbortController();
  const pending = captureCameraFrames(video, 1, controller.signal, () => assert.fail("No photo should be emitted"));
  video.present(1);
  await Promise.resolve();
  controller.abort();
  timeoutCallbacks.at(-1)();
  encodings[0]();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(fallbacks.length, 0);
  assert.equal(canvas.width, 0);
  assert.equal(canvas.height, 0);
});

test("cancellation during synchronous fallback does not emit the encoded photo", async (t) => {
  const video = mockVideo();
  const { canvas } = mockCanvas(t, { failAt: 1 });
  const controller = new AbortController();
  const encode = canvas.toDataURL;
  canvas.toDataURL = (...args) => {
    const dataUrl = encode(...args);
    controller.abort();
    return dataUrl;
  };
  const pending = captureCameraFrames(video, 1, controller.signal, () => assert.fail("No photo should be emitted"));
  video.present(1);
  await assert.rejects(pending, { name: "AbortError" });
});

for (const fallbackFailure of ["throw", "data:,", "data:image/jpeg;base64,", "data:text/plain;base64,YQ==", "data:image/jpeg;base64,!!!!"]) {
  test(`reports both encoding failures when fallback returns ${fallbackFailure}`, async (t) => {
    const video = mockVideo();
    const { canvas, fallbacks } = mockCanvas(t, { failAt: 1, fallbackFailure });
    const pending = captureCameraFrames(video, 1, new AbortController().signal, () => assert.fail("No partial photo"));
    video.present(1);
    await assert.rejects(pending, (error) => {
      assert.match(error.message, /Не удалось сохранить фото/);
      assert.ok(error.cause instanceof AggregateError);
      assert.equal(error.cause.errors.length, 2);
      assert.match(error.cause.errors[0].message, /toBlob/);
      return true;
    });
    assert.equal(fallbacks.length, 1);
    assert.equal(canvas.width, 0);
  });
}

test("reports a security restriction explicitly when both encoders are blocked", async (t) => {
  const video = mockVideo();
  mockCanvas(t, { failAt: 1, blobFailure: "security", fallbackFailure: "security" });
  const pending = captureCameraFrames(video, 1, new AbortController().signal, () => assert.fail("No photo"));
  video.present(1);
  await assert.rejects(pending, (error) => {
    assert.match(error.message, /Браузер запретил/);
    assert.deepEqual(error.cause.errors.map((cause) => cause.name), ["SecurityError", "SecurityError"]);
    return true;
  });
});

for (const fallback of [false, true]) {
  test(`preserves PNG MIME and extension if the ${fallback ? "fallback" : "primary"} encoder returns PNG`, async (t) => {
    const video = mockVideo();
    mockCanvas(t, { mimeType: "image/png", failAt: fallback ? 1 : undefined });
    const pending = captureCameraFrames(video, 1, new AbortController().signal, () => {});
    video.present(1);
    const [file] = await pending;
    assert.equal(file.type, "image/png");
    assert.match(file.name, /\.png$/);
    assert.equal(await file.text(), "frame-1");
  });
}

test("chunked data URL conversion preserves every byte across base64 boundaries", async (t) => {
  const bytes = Uint8Array.from({ length: 80_003 }, (_, index) => index % 256);
  const video = mockVideo();
  mockCanvas(t, { failAt: 1, fallbackBytes: bytes });
  const pending = captureCameraFrames(video, 1, new AbortController().signal, () => {});
  video.present(1);
  const [file] = await pending;
  assert.deepEqual(new Uint8Array(await file.arrayBuffer()), bytes);
});
