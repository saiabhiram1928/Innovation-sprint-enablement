import dotenv from 'dotenv'
dotenv.config({})
import { Agent, run, webSearchTool } from "@openai/agents";
import { z } from 'zod'
import { sendEmail } from './app.ts';
// let instructions: string = "You are a research assistant. Given a search term, you search the web for that term and \
// produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
// words. Capture the main points. Write succintly, no need to have complete sentences or good \
// grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
// essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
// const plannerAgent = new Agent({
//     name: "Planner Agent",
//     instructions: instructions,
//     tools: [webSearchTool()]
// })
// let prompt = "Latest AI Agent frameworks in 2025"

// const result = await run(plannerAgent, prompt)
// console.log("The result of the functions : " + result.finalOutput)


export const webSearchItem = z.object({
    reason: z.string().describe("your reasoning for why this search is imp for this query"),
    query: z.string().describe("The search Item to use for the web search")
})
export const webSearchItemList = z.object({
    searches: z.array(webSearchItem).describe("A list of web searches to perform to best answer the query")
})

export type WebSearchItemListType = z.infer<typeof webSearchItemList>

const searches = 2
let plannerAgentInstructions: string = `You are a helpful research assistant. You will be Given a query, come up with a set of web searches \
to perform to best answer the query. Output  ${searches} terms to query for.`
const plannerAgent = new Agent({
    name: "Planner Agent",
    instructions: plannerAgentInstructions,
    outputType: webSearchItemList,
    model: process.env.MODEL_NAME
})


let writerAgentInstructions: string = "You are a senior researcher tasked with writing a cohesive report for a research query.\n You will be provided with the original query, and some initial research done by a research assistant.\n You should first come up with an outline for the report that describes the structure and  flow of the report. Then, generate the report and return that as your final output.\n The final output should be in markdown format, and it should be lengthy and detailed. Aim  for 5-10 pages of content, at least 1000 words."

const writerAgent = new Agent({
    name: "WriterAgent",
    instructions: writerAgentInstructions,
    model: process.env.MODEL_NAME,
    outputType: z.object({
        short_summary: z.string().describe("A short 2-3 sentence summary of the findings."),
        markdown_reports: z.string().describe("The final Report"),
        follow_up_questions: z.string().describe("suggested topics to research further")
    })
})

let searchAgentInstructions = "You are a research assistant. Given a search term, you search the web for that term and \
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 \
words. Capture the main points. Write succintly, no need to have complete sentences or good \
grammar. This will be consumed by someone synthesizing a report, so it's vital you capture the \
essence and ignore any fluff. Do not include any additional commentary other than the summary itself"

const searchAgent = new Agent({
    name: "Planner Agent",
    instructions: searchAgentInstructions,
    tools: [webSearchTool()],
    model: process.env.MODEL_NAME,
    modelSettings: { toolChoice: "required" },
    outputType: z.object({
        result: z.string()
    })
})

let emailAgentInstructions: string = "  You are able to send a nicely formatted HTML email based on a detailed report. You will be provided with a detailed report.You should use your tool to send one email, providing the report converted into clean, well presented HTML with an appropriate subject line"

const emailAgent = new Agent({
    name: "Report Agent",
    instructions: emailAgentInstructions,
    tools: [sendEmail],
    model: process.env.MODEL_NAME,
})

const plannerAgentRun = async (input: string) => {
    try {
        console.log("==".repeat(20) + "Planning Starts ......" + "==".repeat(20))
        const plans = await run(plannerAgent, input)
        console.log("==".repeat(20) + "Planning Endding ......" + "==".repeat(20))
        return plans.finalOutput;
    } catch (e) {
        console.error("---".repeat(10) + "error occured searching" + "---".repeat(10))
        throw new Error(e);
    }
}


const searchAgentRun = async (searchPlans: WebSearchItemListType) => {
    try {
        console.log("===".repeat(10) + "Search Started...." + "===".repeat(10))
        let tasks = await Promise.all(
            searchPlans.searches.map(async (item) => {
                const searchPrompt = `Search term: ${item.query} \nReason for searching: ${item.reason}`
                let res = await run(searchAgent, searchPrompt)
                return res.finalOutput?.result ?? ""
            })
        )
        console.log("===".repeat(10) + "Search Started...." + "===".repeat(10))
        return tasks;
    } catch (e) {
        console.error("---".repeat(10) + "error in Searching Each Planner Item" + '---'.repeat(10))
        throw new Error(e);
    }
}

const writerAgentRun = async (prompt: string, searchResults: Array<string>) => {
    let input = `Original query: ${prompt} \nSummarized search results: ${searchResults}`
    try {
        console.log("===".repeat(10) + "writing Started...." + "===".repeat(10))
        const results = await run(writerAgent, input)
        console.log("===".repeat(10) + "writing Ended...." + "===".repeat(10))
        return results.finalOutput?.markdown_reports
    } catch (e) {
        console.error("---".repeat(10) + "error in Writer Agent function" + '---'.repeat(10))
        throw new e;
    }
}

const sendEmailAgentRun = async (markdownReport: string) => {
    try {
        console.log("===".repeat(10) + "Email Sending...." + "===".repeat(10))
        const res = await run(emailAgent, markdownReport)
        console.log("===".repeat(10) + "Email Sent...." + "===".repeat(10))
        return res.finalOutput
    } catch (e) {
        console.error("---".repeat(10) + "error in Send Email function" + '---'.repeat(10))
        throw new e;
    }
}

const workFlow = async (prompt: string) => {
    try {
        const plans = await plannerAgentRun(prompt)
        if (!plans) throw new Error("PlanAgent failed")
        const searchResults = await searchAgentRun(plans)
        const markdownReport = await writerAgentRun(prompt, searchResults)
        if (!markdownReport) throw new Error("planner Failed")
        await sendEmailAgentRun(markdownReport)
    } catch (e) {
        console.error(e)
    }
}
let prompt = "Latest AI Agent frameworks in 2025"
await workFlow(prompt)