import json
import os

def export_conversation():
    transcript_path = "/home/ozymandias/.gemini/antigravity-ide/brain/fa8e999b-a9fd-4316-83b9-f6e358d6f000/.system_generated/logs/transcript_full.jsonl"
    output_path = "/home/ozymandias/Desktop/work/kickme/conversation_export.md"
    
    if not os.path.exists(transcript_path):
        print(f"Could not find transcript at {transcript_path}")
        return
        
    print("Exporting conversation to", output_path)
    
    with open(transcript_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as out:
        out.write("# Conversation Export\n\n")
        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            step_type = data.get("type")
            source = data.get("source")
            content = data.get("content", "")
            
            if not content.strip():
                continue
                
            # Filter out system ephemeral messages and empty content
            if step_type == "USER_INPUT":
                # Clean up <USER_REQUEST> tags if they exist
                clean_content = content.replace("<USER_REQUEST>", "").replace("</USER_REQUEST>", "").strip()
                out.write("## User\n\n")
                out.write(clean_content + "\n\n---\n\n")
            elif source == "MODEL" and step_type == "PLANNER_RESPONSE":
                out.write("## Antigravity (Assistant)\n\n")
                out.write(content.strip() + "\n\n---\n\n")
                
    print("Export complete!")

if __name__ == "__main__":
    export_conversation()
