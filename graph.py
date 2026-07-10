# graph.py
from langgraph.graph import StateGraph, END
from state import DebateState
from agents import proposer_agent, challenger_agent, devil_agent, judge_agent
from config import NUM_ROUNDS

def should_continue(state: DebateState) -> str:
    if state["round_number"] <= NUM_ROUNDS:
        return "continue"
    else:
        return "end"

def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("Proposer", proposer_agent)
    graph.add_node("Challenger", challenger_agent)
    graph.add_node("Devil", devil_agent)
    graph.add_node("Judge", judge_agent)

    graph.set_entry_point("Proposer")

    graph.add_edge("Proposer", "Challenger")
    graph.add_edge("Challenger", "Devil")

    graph.add_conditional_edges(
        "Devil",
        should_continue,
        {
            "continue": "Proposer",
            "end": "Judge"
        }
    )

    graph.add_edge("Judge", END)

    return graph.compile()