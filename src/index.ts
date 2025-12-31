import { AppServer, AppSession } from '@mentra/sdk';
import { GoogleGenAI } from '@google/genai'
import { config } from './config';

const ai = new GoogleGenAI({ apiKey: config.GEMINI_API_KEY });

async function extractPersonInfo(conversation: string): Promise<{
    name?: string;
    workplace?: string;
    context?: string;
    details?: string;
}> {
    const prompt = `
    I just met someone and had the following conversation after asking, "What's your name?":

    "${conversation}"
    Extract any information about this person. Return ONLY valid JSON (no markdown, no backticks):
    {
        "name": "their name or null if not mentioned",
        "workplace": "where they work/study/major or null",
        "context": "how/where we met or null",
        "details": "any other notable info or null"
    }`;

    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        const text = response.text ?? '';
        
        try {
            return JSON.parse(text);
        } catch (parseError) {
            console.error('Failed to parse Gemini response:', text);
            return {};
        }
    } catch (apiError) {
        console.error('Gemini API error:', apiError);
        return {};
    }
}

class MentraOSApp extends AppServer {
    private conversationBuffer: string[] = []
    private isCollecting = false;
    private capturedPhoto: Buffer | null = null;

    constructor() {
        super({
            packageName: config.PACKAGE_NAME,
            apiKey: config.MENTRAOS_API_KEY,
            port: config.PORT,
        })
    }

    protected override async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
        session.layouts.showTextWall("App has started")
        console.log("App has begun running");

        session.events.onTranscription(async (data) => {
            console.log('Transcription received:', data.text, 'isFinal:', data.isFinal);

            // Handle conversation collection
            if (this.isCollecting) {
                if (data.isFinal) {
                    this.conversationBuffer.push(data.text);
                    console.log('📝 Buffered:', data.text);

                    const text = data.text.toLowerCase();

                    // Check for farewell phrase
                    if (text.includes("nice to meet you") || text.includes("nice meeting you") || text.includes("catch you later")) {
                        this.isCollecting = false;
                        const fullConversation = this.conversationBuffer.join(' ');
                        console.log('📝 Farewell detected! Conversation:', fullConversation);

                        const personInfo = await extractPersonInfo(fullConversation);
                        console.log('Extracted info:', personInfo);

                        const base64Image = this.capturedPhoto?.toString('base64');

                        if (!base64Image) {
                            console.error('❌ No photo was captured');
                            await session.audio.speak("Visage couldn't capture a photo");
                            return;
                        }

                        try {
                            const response = await fetch(`${config.BACKEND_URL}/api/workflow1/first-meeting`, {
                                method: "POST",
                                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                                body: new URLSearchParams({
                                    image_data: base64Image,
                                    name: personInfo.name || '',
                                    conversation_context: `${personInfo.workplace || ''} ${personInfo.context || ''} ${personInfo.details || ''}`.trim()
                                })
                            });

                            if (!response.ok) {
                                const errorText = await response.text();
                                console.error(`❌ Backend error: ${response.status} — ${errorText}`);
                                await session.audio.speak("Visage couldn't save that information");
                                return;
                            }

                            const result = await response.json();
                            console.log('✅ Saved to database:', result);
                            await session.audio.speak(`Visage will remember ${personInfo.name || 'them'}`);
                        } catch (fetchError) {
                            console.error('❌ Failed to connect to backend:', fetchError);
                            await session.audio.speak("Visage couldn't save that information");
                        }

                        this.conversationBuffer = [];
                        return;
                    }
                }
                return;
            }

            if (!data.isFinal) return;

            const command = data.text.toLowerCase();
            console.log('🎯 Processing command:', command);



            // Trigger phrase to start collecting
            if (command.includes("hey, what's your name") || command.includes("hey, whats your name")) {
                try {
                    this.isCollecting = true;
                    this.conversationBuffer = [];

                    console.log('Starting photo request...');
                    session.camera.requestPhoto({
                        size: 'small',
                        compress: 'medium'
                    })
                        .then(async photo => {
                            console.log(`Photo captured: ${photo.filename}`)
                            this.capturedPhoto = photo.buffer
                            await session.audio.speak("Photo Captured")
                        })
                        .catch(async err => {
                            console.error("Photo failed", err);
                            await session.audio.speak("Camera isn't available right now");
                        });

                    // Timeout after 20 seconds
                    setTimeout(async () => {
                        if (this.isCollecting) {
                            this.isCollecting = false;
                            const fullConversation = this.conversationBuffer.join(' ');
                            console.log('📝 Timeout! Conversation:', fullConversation);

                            const personInfo = await extractPersonInfo(fullConversation);
                            console.log('📋 Extracted info:', personInfo);

                            const base64Image = this.capturedPhoto?.toString('base64');

                            if (!base64Image) {
                                console.error('❌ No photo was captured');
                                await session.audio.speak("Visage couldn't capture a photo");
                                this.conversationBuffer = [];
                                return;
                            }

                            try {
                                const response = await fetch(`${config.BACKEND_URL}/api/workflow1/first-meeting`, {
                                    method: "POST",
                                    headers: {
                                        'Content-Type': 'application/x-www-form-urlencoded',
                                    },
                                    body: new URLSearchParams({
                                        image_data: base64Image,
                                        name: personInfo.name || '',
                                        conversation_context: `${personInfo.workplace || ''} ${personInfo.context || ''} ${personInfo.details || ''}`.trim()
                                    })
                                });

                                if (!response.ok) {
                                    const errorText = await response.text();
                                    console.error(`❌ Backend error: ${response.status} - ${response.statusText}`);
                                    await session.audio.speak("Visage failed to save that information");
                                    return;
                                }
                                
                                const result = await response.json();
                                console.log('✅ Saved to database:', result);
                                await session.audio.speak(`Visage will remember ${personInfo.name || 'them'}`);
                            } catch (fetchError) {
                                console.error('❌ Failed to connect to backend:', fetchError);
                                await session.audio.speak("Visage failed to save that information");
                            }

                            this.conversationBuffer = [];
                            this.capturedPhoto = null;
                        }
                    }, 20000);

                } catch (err) {
                    console.error('Failed to capture photo', err);
                }
            }
        });

        await session.audio.speak("Visage has started");
    }
}

const app = new MentraOSApp();
app.start().catch(console.error);
