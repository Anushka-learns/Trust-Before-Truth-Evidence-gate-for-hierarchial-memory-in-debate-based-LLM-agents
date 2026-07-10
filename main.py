# main.py
import uuid
from graph import build_graph
from memory import retrieve_all_memories, save_all_memories

def main():
    app = build_graph()

    question = "for a student what api key is the best to use?"
    
    debate_id = question.replace(" ", "_")[:50]

   
    memories = retrieve_all_memories(question, debate_id)

    initial_state = {
        "topic": question,
        "messages": [],
        "structured_messages": [],
        "round_number": 1,
        "last_message": f"The question being debated is: {question}",
        "last_structured": {},
        "verdict": "",
        "verdict_structured": {},
        "retrieved_memories": memories,
        "contradiction_count": 0
    }

    print("=" * 60)
    print(f"DEBATE QUESTION: {question}")
    print(f"MEMORIES RETRIEVED: {sum(len(v) for v in memories.values())}")
    print("=" * 60)

    
    final_state = app.invoke(initial_state)

    
    transcript = "\n".join([f"{role}: {msg}" for role, msg in final_state['messages']])
    save_all_memories(
        question=question,
        transcript=transcript,
        verdict_structured=final_state['verdict_structured'],
        debate_id=debate_id,
        threshold=7.0
    )

    
    print("\n" + "=" * 60)
    print("DEBATE COMPLETE — FINAL SUMMARY")
    print("=" * 60)
    print(f"Final Answer:        {final_state['verdict']}")
    print(f"Confidence:          {final_state['verdict_structured'].get('confidence', 0)}/10")
    print(f"Contradiction Score: {final_state['verdict_structured'].get('contradiction_score', 0)}/10")
    print(f"Winning Side:        {final_state['verdict_structured'].get('winning_side', '')}")
    print(f"Key Fact:            {final_state['verdict_structured'].get('key_fact', '')}")

if __name__ == "__main__":
    main()
   