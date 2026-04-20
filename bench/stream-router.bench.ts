import { bench, group } from "mitata";
import { StreamRouter } from "../src/inference/stream-router.js";

group("emit", () => {
  bench("1 consumer, 100 messages", () => {
    const router = new StreamRouter<string>();
    router.createConsumer("c0");
    for (let i = 0; i < 100; i++) {
      router.emit("c0", `msg-${i}`);
    }
    router.close("c0");
  });

  bench("5 consumers, 100 messages each", () => {
    const router = new StreamRouter<string>();
    for (let c = 0; c < 5; c++) {
      router.createConsumer(`c${c}`);
    }
    for (let c = 0; c < 5; c++) {
      for (let i = 0; i < 100; i++) {
        router.emit(`c${c}`, `msg-${i}`);
      }
    }
    for (let c = 0; c < 5; c++) {
      router.close(`c${c}`);
    }
  });
});

group("round-trip (emit + consume)", () => {
  bench("1 consumer, 50 messages", async () => {
    const router = new StreamRouter<string>();
    const consumer = router.createConsumer("c0");
    for (let i = 0; i < 50; i++) {
      router.emit("c0", `msg-${i}`);
    }
    router.close("c0");
    const tokens: string[] = [];
    for await (const token of consumer) tokens.push(token);
  });
});
