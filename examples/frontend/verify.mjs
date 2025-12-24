import { StreamProcessor } from "@tanstack/ai";

const endpoint = process.env.SSE_ENDPOINT || "http://127.0.0.1:8000/chat/stream";
const payload = {
  messages: [{ role: "user", content: "Say hello in one short sentence." }],
};

async function* sseDataLines(response) {
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      if (value) {
        buffer += decoder.decode(value, { stream: true });
      }
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const lines = frame.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              return;
            }
            yield data;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function chunkIterator(response) {
  return (async function* () {
    for await (const data of sseDataLines(response)) {
      yield JSON.parse(data);
    }
  })();
}

async function main() {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }

  const processor = new StreamProcessor({
    events: {
      onTextUpdate: (_messageId, content) => {
        process.stdout.write(`\rStreaming: ${content}`);
      },
      onStreamEnd: (message) => {
        process.stdout.write("\n");
        const textParts = message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.content)
          .join("");
        console.log("Final assistant message:", textParts);
      },
    },
  });

  let sawChunk = false;
  processor.startAssistantMessage();
  await processor.process(chunkIterator(response));
  sawChunk = processor.getMessages().length > 0;

  if (!sawChunk) {
    throw new Error("No chunks received from SSE stream.");
  }

  const messages = processor.getMessages();
  if (messages.length === 0) {
    throw new Error("StreamProcessor has no messages after stream end.");
  }

  console.log("SSE parsing verified with StreamProcessor.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
