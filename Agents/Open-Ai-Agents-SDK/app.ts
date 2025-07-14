import dotnev from 'dotenv';
dotnev.config()
import { Agent, run, tool } from '@openai/agents';
import nodemailer from 'nodemailer'
const transporter = nodemailer.createTransport({
    host: 'smtp.ethereal.email',
    port: 587,
    auth: {
        user: 'vance72@ethereal.email',
        pass: 'HrA4MesvXEdMHnbJAa'
    }
})


interface SendEmail {
    status: string,
    sucess: boolean
}
const sendEmail = tool({
    name: "sendEmail",
    description: "Sends an email to the given address.",
    parameters: {
        type: "object",
        properties: {
            subject: {
                type: 'string',
                description: "The subject for the email to send the email"
            },
            body: {
                type: "string",
                description: "The email address or content to send the email to"
            }
        },
        required: ["body", "subject"],
        additionalProperties: false
    },
    strict: true,
    execute: async (input: any): Promise<SendEmail> => {
        const { body, subject } = input as { body: string, subject: string };
        console.log("Function sendEmail has been called with body: " + body);
        try {
            const info = await transporter.sendMail({
                from: "vance <vance72@ethereal.email>",
                to: "test@gmail.com",
                subject: subject,
                html: body
            })
            return {
                status: `Email has been sent with ${info.messageId}`,
                sucess: true
            };
        } catch (error) {
            console.error("Error while sending the mail")
            throw error;
        }
    }
});
let instructions_1: string = "You are a sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI.\
You write professional, serious cold emails."
let instructions_2: string = "You are a humorous, engaging sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write witty, engaging cold emails that are likely to get a response."
let instructions_3: string = "You are a busy sales agent working for ComplAI, \
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
You write concise, to the point cold emails."

const email_agent_1 = new Agent({
    name: "email-agent-01",
    model: process.env.MODEL_NAME,
    instructions: instructions_1
})

const email_agent_2 = new Agent({
    name: "email-agent-02",
    model: process.env.MODEL_NAME,
    instructions: instructions_2
})
const email_agent_3 = new Agent({
    name: "email-agent-03",
    model: process.env.MODEL_NAME,
    instructions: instructions_3
})
const email_agent_tool_1 = email_agent_1.asTool({
    toolName: "sales_agent1",
    toolDescription: "Write cold email",
})
const email_agent_tool_2 = email_agent_2.asTool({
    toolName: "sales_agent2",
    toolDescription: "Write cold email",
})
const email_agent_tool_3 = email_agent_3.asTool({
    toolName: "sales_agent3",
    toolDescription: "Write cold email",
})
let tools = [email_agent_tool_1, email_agent_tool_2, email_agent_tool_3]
let final_instructions: string = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales_agent tools once before choosing the best one. \
You pick the single best email and use the send_email tool to send the best email (and only the best email) to the user. Again you have to pick the best mail shouldn't send the the three mail's"



let subject_instructions: string = "You can write a subject for a cold sales email. \
You are given a message and you need to write a subject for an email that is likely to get a response"
let html_instructions: string = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

const subject_tool = new Agent({
    name: "subject_agent",
    instructions: subject_instructions,
    model: process.env.MODEL_NAME
}).asTool({
    toolName: "subject_tool",
    toolDescription: "Write subject from the given mail"
})
const html_convertor_tool = new Agent({
    name: "html_convertor_agent",
    instructions: html_instructions,
    model: process.env.MODEL_NAME
}).asTool({
    toolName: "html_convertor_tool",
    toolDescription: "Converts the body of the email form text to html"
})

let new_tools = [subject_tool, html_convertor_tool, sendEmail]

let email_sender_instructions: string = "You are an email formatter and sender. You receive the body of an email to be sent. \
You first use the subject_tool tool to write a subject for the email, then use the html_convertor_tool tool to convert the body to HTML. \
Finally, you use the sendEmail tool to send the email with the subject and HTML body. The sendEmail tool accepts the subject and body as parameters "

const email_sender_agent: Agent = new Agent({
    name: "email_sender_agent",
    instructions: email_sender_instructions,
    tools: new_tools,
    model: process.env.MODEL_NAME,
    handoffDescription: "Convert an email to HTML and send it"
})
const handoffs: Array<Agent> = [email_sender_agent]
const manager_instructions = "You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. \
You never generate sales emails yourself; you always use the tools. \
You try all 3 sales agent tools at least once before choosing the best one. \
You can use the tools multiple times if you're not satisfied with the results from the first try. \
You select the single best email using your own judgement of which email will be most effective. \
After picking the email, you handoff to the Email Manager agent to format and send the email."

const managerAgent: Agent = new Agent({
    name: "manager_agent",
    instructions: manager_instructions,
    tools: tools,
    model: process.env.MODEL_NAME,
    handoffs: handoffs
})

const prompt: string = "Send out a cold sales email addressed to Dear CEO from Alice to test@gmail.com"

const result = await run(managerAgent, prompt)

console.log("=".repeat(50))
console.log(result)
console.log("=".repeat(50))



