import { Agent, ModelBehaviorError, OpenAIChatCompletionsModel, run, Runner } from '@openai/agents'
import dotenv from 'dotenv'
import openai from 'openai'
dotenv.config()

const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

const gemmnai_client = new openai({
    apiKey: process.env.GOOGLE_API_KEY,
    baseURL: GEMINI_BASE_URL,
})
const deepseek_model = new OpenAIChatCompletionsModel(
    gemmnai_client,
    "gemini-2.0-flash"
)

const agent = new Agent({
    name: "Gemini Agent",
    instructions: "You are helpful gemini agent ?",
    model: deepseek_model
})

const result = await run(agent, "Which model are you?")
console.log("==".repeat(50))
console.log(result.finalOutput)
console.log("==".repeat(50))