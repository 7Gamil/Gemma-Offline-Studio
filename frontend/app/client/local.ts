import { ChatOptions, LLMApi, ModelRecord } from "./api";

export class LocalApi extends LLMApi {
    private controller = new AbortController();
    
    async chat(options: ChatOptions): Promise<void> {
        const {
            messages,
            config,
            onUpdate,
            onFinish,
            onError,
        } = options;
        
        const controller = new AbortController();
        this.controller = controller;
        
        try {
            const response = await fetch("http://127.0.0.1:8000/v1/chat/completions", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: config.model,
                    messages: messages,
                    stream: false, // Set to false to receive a single JSON response
                }),
                signal: controller.signal,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const responseText = data.choices[0]?.message?.content || "";
            
            if (onUpdate) {
                onUpdate(responseText, responseText);
            }
            
            onFinish(responseText, "stop");
        } catch (e) {
            if (onError) {
                onError(e as Error);
            }
        }
    }
    
    async abort(): Promise<void> {
        this.controller.abort();
    }

    async tts(text: string): Promise<void> {
        const cleanText = text.replace(/[*#`]/g, "");
        if (cleanText.trim().length === 0) {
            return;
        }
        const response = await fetch("http://127.0.0.1:8000/v1/tts", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: cleanText,
                voice: "af_heart"
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(new Blob([audioBlob], { type: "audio/wav" }));
        const audio = new Audio(audioUrl);
        try {
            await audio.play();
        } catch (e) {
            console.error("Failed to play audio:", e);
        }
    }
    
    async models(): Promise<ModelRecord[]> {
        const response = await fetch("http://127.0.0.1:8000/v1/models");
        const data = await response.json();
        return data.data.map((model: any) => ({
            name: model.id,
            display_name: model.id,
            family: "gemma",
        }));
    }
}
