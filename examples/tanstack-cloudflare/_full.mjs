import { AgUiThreadRuntimeCore } from "/Users/alex/proj/subs/examples/tanstack-cloudflare/../../examples/node_modules/.pnpm/@assistant-ui+react-ag-ui@0.0.34_@assistant-ui+store@0.2.13_@assistant-ui+tap@0.5.14_@t_870b04ba7fad871a817830a61d7d89e8/node_modules/@assistant-ui/react-ag-ui/dist/runtime/AgUiThreadRuntimeCore.js";
import { HttpAgent } from "/Users/alex/proj/subs/examples/tanstack-cloudflare/../../examples/node_modules/.pnpm/@ag-ui+client@0.0.54/node_modules/@ag-ui/client/dist/index.mjs";
import { ToolInvocationTracker } from "/Users/alex/proj/subs/examples/tanstack-cloudflare/../../examples/node_modules/.pnpm/@assistant-ui+core@0.2.9_@assistant-ui+store@0.2.13_@assistant-ui+tap@0.5.14_@types+rea_da23f2cc54da433397dc1c18bdd44a16/node_modules/@assistant-ui/core/dist/runtimes/tool-invocations/ToolInvocationTracker.js";
const agent = new HttpAgent({ url:"http://localhost:9000/api/client/ag-ui/agents/weather-agent/run", headers:{Authorization:`Bearer ${process.argv[2]}`}, fetch:(u,i)=>fetch(u,i) });
const tracker = new ToolInvocationTracker(
  () => ({ get_user_timezone: { execute: async () => { console.log("  [EXECUTE] RAN"); return { timezone:"America/Los_Angeles" }; } } }),
  { onResult: (c)=>console.log("  [onResult]", c.toolName, JSON.stringify(c.result)), onStatusesChange: ()=>{} }
);
let core;
const feed = () => { try { tracker.setState({ messages: core.getMessages(), isRunning: core.isRunning?.() ?? false }); } catch(e){ console.log("feed err", e.message); } };
core = new AgUiThreadRuntimeCore({ agent, logger:{debug:()=>{},error:()=>{}}, showThinking:true, notifyUpdate: feed });
core.attachRuntime({ thread: { getModelContext: () => ({ tools:{} }) } });
await core.append({ role:"user", content:[{type:"text",text:"what timezone am i in?"}], startRun:true });
await new Promise(r=>setTimeout(r,9000));
console.log("=== final assistant content ===");
const a=[...core.getMessages()].reverse().find(m=>m.role==="assistant"&&m.content?.some(p=>p.type==="text"&&p.text));
console.log(JSON.stringify(a?.content?.map(p=>p.type)));
process.exit(0);
