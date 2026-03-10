# Conversational AI

## Table of Contents

- [Introduction](#introduction)
- [Conversation State Management](#conversation-state-management)
- [Context Tracking](#context-tracking)
- [Handling Clarifications and Corrections](#handling-clarifications-and-corrections)
- [Personality and Consistency](#personality-and-consistency)
- [Memory Systems](#memory-systems)
- [Safety and Moderation](#safety-and-moderation)
- [Multi-Turn Dialogue Patterns](#multi-turn-dialogue-patterns)
- [Production Deployment](#production-deployment)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Conversational AI enables natural multi-turn dialogue between humans and machines. Unlike single-turn question answering, conversational systems maintain context across interactions, remember past exchanges, and adapt responses based on the full dialogue history. Modern conversational AI leverages large language models (LLMs) to generate human-like responses while managing complex state and context.

```
Conversational AI Architecture:

┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  "What's the weather?"  →  "I can help! Which city?"           │
│  "Seattle"              →  "Seattle: 65°F, partly cloudy"       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Conversation Manager                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐            │
│  │   State    │  │  Context   │  │    Memory    │            │
│  │  Tracker   │  │  Manager   │  │    Store     │            │
│  └────────────┘  └────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LLM + Tools Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐  │
│  │   LLM    │  │  Intent  │  │  Response  │  │  Safety    │  │
│  │  Engine  │  │  Parser  │  │ Generator  │  │   Filter   │  │
│  └──────────┘  └──────────┘  └────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components**:

- **State Management**: Track conversation status and flow
- **Context Tracking**: Maintain dialogue history and references
- **Intent Recognition**: Understand what the user wants
- **Response Generation**: Create natural, contextual replies
- **Memory Systems**: Store and retrieve user information
- **Safety & Moderation**: Filter harmful content
- **Personality Control**: Maintain consistent character

**Use Cases**:

- **Customer Support**: 24/7 automated assistance with escalation
- **Virtual Assistants**: Task automation and information retrieval
- **Education**: Interactive tutoring and learning companions
- **Healthcare**: Patient triage and health information
- **Entertainment**: Interactive storytelling and gaming
- **E-commerce**: Shopping assistance and recommendations

This guide provides comprehensive patterns and implementations for building production-ready conversational AI systems.

## Conversation State Management

State management is the foundation of conversational AI. It tracks the current status of the conversation, manages transitions between topics, and maintains the flow of dialogue.

### Conversation States

```python
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

class ConversationState(Enum):
    """States a conversation can be in."""
    INITIALIZED = "initialized"
    GREETING = "greeting"
    GATHERING_INFO = "gathering_info"
    CLARIFYING = "clarifying"
    PROCESSING = "processing"
    RESPONDING = "responding"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ERROR = "error"
    CLOSING = "closing"
    ENDED = "ended"

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    turn_id: int
    timestamp: datetime
    user_message: str
    bot_response: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSession:
    """Complete conversation session data."""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    state: ConversationState
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def duration_seconds(self) -> float:
        return (self.last_activity - self.start_time).total_seconds()
```

### State Manager Implementation

```python
import uuid
from collections import defaultdict
from typing import Callable, Optional

class StateTransition:
    """Defines a state transition with conditions."""

    def __init__(
        self,
        from_state: ConversationState,
        to_state: ConversationState,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition or (lambda session: True)
        self.action = action or (lambda session: None)

    def can_transition(self, session: ConversationSession) -> bool:
        """Check if transition is allowed."""
        return session.state == self.from_state and self.condition(session)

    def execute(self, session: ConversationSession) -> None:
        """Execute the transition."""
        self.action(session)
        session.state = self.to_state

class ConversationStateManager:
    """Manages conversation state and transitions."""

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.transitions: List[StateTransition] = []
        self._setup_default_transitions()

    def _setup_default_transitions(self):
        """Setup common state transitions."""
        # Greeting → Gathering Info
        self.add_transition(
            ConversationState.GREETING,
            ConversationState.GATHERING_INFO,
            condition=lambda s: s.turn_count >= 1
        )

        # Gathering Info → Clarifying (when unclear)
        self.add_transition(
            ConversationState.GATHERING_INFO,
            ConversationState.CLARIFYING,
            condition=lambda s: s.metadata.get('needs_clarification', False)
        )

        # Clarifying → Gathering Info (after clarification)
        self.add_transition(
            ConversationState.CLARIFYING,
            ConversationState.GATHERING_INFO,
            condition=lambda s: s.metadata.get('clarification_provided', False)
        )

        # Gathering Info → Processing (when complete)
        self.add_transition(
            ConversationState.GATHERING_INFO,
            ConversationState.PROCESSING,
            condition=lambda s: s.metadata.get('info_complete', False)
        )

        # Processing → Responding
        self.add_transition(
            ConversationState.PROCESSING,
            ConversationState.RESPONDING,
            condition=lambda s: s.metadata.get('processing_done', False)
        )

        # Responding → Awaiting Confirmation
        self.add_transition(
            ConversationState.RESPONDING,
            ConversationState.AWAITING_CONFIRMATION,
            condition=lambda s: s.metadata.get('needs_confirmation', False)
        )

        # Any state → Error
        for state in ConversationState:
            if state != ConversationState.ERROR:
                self.add_transition(
                    state,
                    ConversationState.ERROR,
                    condition=lambda s: s.metadata.get('has_error', False)
                )

        # Any state → Closing
        for state in ConversationState:
            if state != ConversationState.CLOSING:
                self.add_transition(
                    state,
                    ConversationState.CLOSING,
                    condition=lambda s: s.metadata.get('user_wants_to_end', False)
                )

    def add_transition(
        self,
        from_state: ConversationState,
        to_state: ConversationState,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None
    ):
        """Add a state transition."""
        transition = StateTransition(from_state, to_state, condition, action)
        self.transitions.append(transition)

    def create_session(self, user_id: str) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            state=ConversationState.INITIALIZED
        )
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def add_turn(
        self,
        session: ConversationSession,
        user_message: str,
        bot_response: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0
    ) -> ConversationTurn:
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            turn_id=len(session.turns),
            timestamp=datetime.now(),
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            entities=entities or {},
            confidence=confidence
        )
        session.turns.append(turn)
        session.last_activity = datetime.now()
        return turn

    def update_state(self, session: ConversationSession) -> bool:
        """Update session state based on transitions."""
        for transition in self.transitions:
            if transition.can_transition(session):
                transition.execute(session)
                return True
        return False

    def set_metadata(self, session: ConversationSession, key: str, value: Any):
        """Set metadata on the session."""
        session.metadata[key] = value

    def get_metadata(self, session: ConversationSession, key: str, default: Any = None) -> Any:
        """Get metadata from the session."""
        return session.metadata.get(key, default)

    def end_session(self, session: ConversationSession):
        """End a conversation session."""
        session.state = ConversationState.ENDED
        session.metadata['end_time'] = datetime.now()

# Example usage
def example_state_management():
    """Demonstrate state management."""
    manager = ConversationStateManager()

    # Create a session
    session = manager.create_session(user_id="user123")
    print(f"Session created: {session.session_id}")
    print(f"Initial state: {session.state}")

    # Simulate greeting
    session.state = ConversationState.GREETING
    manager.add_turn(
        session,
        user_message="Hello!",
        bot_response="Hi! How can I help you today?",
        intent="greeting",
        confidence=0.95
    )

    # Transition to gathering info
    manager.update_state(session)
    print(f"After greeting: {session.state}")

    # Gather information
    manager.add_turn(
        session,
        user_message="I need help with my order",
        bot_response="I can help with that. What's your order number?",
        intent="order_inquiry",
        confidence=0.88
    )

    # Mark as needing clarification
    manager.set_metadata(session, 'needs_clarification', True)
    manager.update_state(session)
    print(f"After unclear response: {session.state}")

    # Provide clarification
    manager.set_metadata(session, 'needs_clarification', False)
    manager.set_metadata(session, 'clarification_provided', True)
    manager.update_state(session)
    print(f"After clarification: {session.state}")

    print(f"\nTotal turns: {session.turn_count}")
    print(f"Duration: {session.duration_seconds:.2f} seconds")

if __name__ == "__main__":
    example_state_management()
```

### State Persistence

```python
import pickle
from pathlib import Path

class ConversationStateStore:
    """Persistent storage for conversation states."""

    def __init__(self, storage_path: str = "./conversation_states"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

    def save_session(self, session: ConversationSession):
        """Save a session to disk."""
        file_path = self.storage_path / f"{session.session_id}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(session, f)

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a session from disk."""
        file_path = self.storage_path / f"{session_id}.pkl"
        if not file_path.exists():
            return None
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def delete_session(self, session_id: str):
        """Delete a session from disk."""
        file_path = self.storage_path / f"{session_id}.pkl"
        if file_path.exists():
            file_path.unlink()

    def list_sessions(self) -> List[str]:
        """List all stored session IDs."""
        return [f.stem for f in self.storage_path.glob("*.pkl")]
```

## Context Tracking

Context tracking maintains the conversation history, resolves references, and tracks important information across turns.

### Context Manager

```python
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

class ContextWindow:
    """Manages a sliding window of conversation context."""

    def __init__(self, max_turns: int = 10, max_tokens: int = 2000):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the context window."""
        self.turns.append(turn)

    def get_context_string(self, include_intents: bool = False) -> str:
        """Get formatted context string."""
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_message}")
            if include_intents and turn.intent:
                lines.append(f"[Intent: {turn.intent}]")
            lines.append(f"Assistant: {turn.bot_response}")
        return "\n".join(lines)

    def get_recent_turns(self, n: int = 3) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return list(self.turns)[-n:]

    def clear(self):
        """Clear the context window."""
        self.turns.clear()

class EntityTracker:
    """Tracks entities mentioned in the conversation."""

    def __init__(self):
        self.entities: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # Maps entity type to list of (value, turn_id) tuples

    def add_entity(self, entity_type: str, value: str, turn_id: int):
        """Add an entity mention."""
        self.entities[entity_type].append((value, turn_id))

    def get_latest_entity(self, entity_type: str) -> Optional[str]:
        """Get the most recent entity of a type."""
        if entity_type in self.entities and self.entities[entity_type]:
            return self.entities[entity_type][-1][0]
        return None

    def get_all_entities(self, entity_type: str) -> List[str]:
        """Get all entities of a type."""
        return [value for value, _ in self.entities.get(entity_type, [])]

    def has_entity(self, entity_type: str) -> bool:
        """Check if entity type has been mentioned."""
        return entity_type in self.entities and len(self.entities[entity_type]) > 0

class ReferenceResolver:
    """Resolves pronouns and references in conversation."""

    def __init__(self):
        self.pronouns = {
            'it', 'this', 'that', 'these', 'those',
            'he', 'she', 'they', 'him', 'her', 'them'
        }
        self.last_subject: Optional[str] = None
        self.last_object: Optional[str] = None

    def contains_reference(self, text: str) -> bool:
        """Check if text contains a reference."""
        words = text.lower().split()
        return any(word in self.pronouns for word in words)

    def resolve_reference(self, text: str, entities: EntityTracker) -> str:
        """Attempt to resolve references in text."""
        resolved = text

        # Simple pronoun resolution
        if self.contains_reference(text.lower()):
            # Try to resolve "it" with last mentioned object/product
            if 'it' in text.lower() and self.last_object:
                resolved = resolved.replace('it', self.last_object)
                resolved = resolved.replace('It', self.last_object.capitalize())

        return resolved

    def update_references(self, entities: Dict[str, Any]):
        """Update reference tracking with new entities."""
        if 'product' in entities:
            self.last_object = entities['product']
        if 'person' in entities:
            self.last_subject = entities['person']

class ContextTracker:
    """Comprehensive context tracking for conversations."""

    def __init__(self, max_turns: int = 10):
        self.context_window = ContextWindow(max_turns=max_turns)
        self.entity_tracker = EntityTracker()
        self.reference_resolver = ReferenceResolver()
        self.active_topic: Optional[str] = None
        self.topic_history: List[str] = []
        self.user_preferences: Dict[str, Any] = {}

    def add_turn(self, turn: ConversationTurn):
        """Add a turn and update context."""
        # Add to context window
        self.context_window.add_turn(turn)

        # Track entities
        for entity_type, value in turn.entities.items():
            if isinstance(value, list):
                for v in value:
                    self.entity_tracker.add_entity(entity_type, v, turn.turn_id)
            else:
                self.entity_tracker.add_entity(entity_type, value, turn.turn_id)

        # Update references
        self.reference_resolver.update_references(turn.entities)

        # Track topics
        if turn.intent and turn.intent != self.active_topic:
            if self.active_topic:
                self.topic_history.append(self.active_topic)
            self.active_topic = turn.intent

    def get_context_for_llm(self, include_system_prompt: bool = True) -> str:
        """Get formatted context for LLM."""
        parts = []

        if include_system_prompt:
            parts.append("You are a helpful AI assistant. Here is the conversation history:")

        # Add conversation history
        parts.append(self.context_window.get_context_string())

        # Add entity context
        if self.entity_tracker.entities:
            parts.append("\nKey entities mentioned:")
            for entity_type, mentions in self.entity_tracker.entities.items():
                latest = mentions[-1][0] if mentions else None
                if latest:
                    parts.append(f"- {entity_type}: {latest}")

        # Add topic context
        if self.active_topic:
            parts.append(f"\nCurrent topic: {self.active_topic}")

        return "\n".join(parts)

    def resolve_user_input(self, user_input: str) -> str:
        """Resolve references in user input."""
        return self.reference_resolver.resolve_reference(user_input, self.entity_tracker)

    def get_entity(self, entity_type: str) -> Optional[str]:
        """Get the latest entity of a type."""
        return self.entity_tracker.get_latest_entity(entity_type)

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.user_preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.user_preferences.get(key, default)

    def clear_context(self):
        """Clear all context."""
        self.context_window.clear()
        self.entity_tracker = EntityTracker()
        self.active_topic = None
        self.topic_history.clear()

# Example usage
def example_context_tracking():
    """Demonstrate context tracking."""
    tracker = ContextTracker(max_turns=5)

    # Turn 1
    turn1 = ConversationTurn(
        turn_id=0,
        timestamp=datetime.now(),
        user_message="I'm looking for a laptop",
        bot_response="I can help you find a laptop. What's your budget?",
        intent="product_search",
        entities={'product': 'laptop'}
    )
    tracker.add_turn(turn1)

    # Turn 2
    turn2 = ConversationTurn(
        turn_id=1,
        timestamp=datetime.now(),
        user_message="Around $1000",
        bot_response="Great! Are you looking for gaming or productivity?",
        intent="product_search",
        entities={'budget': '1000', 'currency': 'USD'}
    )
    tracker.add_turn(turn2)

    # Turn 3 with reference
    user_input_with_ref = "I need it for gaming"
    resolved_input = tracker.resolve_user_input(user_input_with_ref)
    print(f"Original: {user_input_with_ref}")
    print(f"Resolved: {resolved_input}")

    turn3 = ConversationTurn(
        turn_id=2,
        timestamp=datetime.now(),
        user_message=resolved_input,
        bot_response="Perfect! Here are some gaming laptops under $1000...",
        intent="product_search",
        entities={'purpose': 'gaming'}
    )
    tracker.add_turn(turn3)

    # Get context for LLM
    print("\n" + "="*60)
    print("Context for LLM:")
    print("="*60)
    print(tracker.get_context_for_llm())

    # Check entities
    print("\n" + "="*60)
    print(f"Latest product: {tracker.get_entity('product')}")
    print(f"Budget: {tracker.get_entity('budget')}")
    print(f"Active topic: {tracker.active_topic}")

if __name__ == "__main__":
    example_context_tracking()
```

## Handling Clarifications and Corrections

Conversational AI must handle unclear inputs, ask clarifying questions, and allow users to correct mistakes.

### Clarification Handler

```python
from typing import List, Optional, Callable
import re

class ClarificationStrategy:
    """Strategy for handling different types of clarifications."""

    @staticmethod
    def missing_entity(entity_name: str, examples: Optional[List[str]] = None) -> str:
        """Generate question for missing entity."""
        question = f"Could you please specify the {entity_name}?"
        if examples:
            question += f" For example: {', '.join(examples)}"
        return question

    @staticmethod
    def ambiguous_intent(possible_intents: List[str]) -> str:
        """Generate question for ambiguous intent."""
        options = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(possible_intents)])
        return f"I'm not sure what you mean. Did you want to:\n{options}\nPlease select a number."

    @staticmethod
    def low_confidence(original_understanding: str) -> str:
        """Generate confirmation question for low confidence."""
        return f"Just to confirm, you want to {original_understanding}. Is that correct?"

    @staticmethod
    def multiple_values(entity_name: str, values: List[str]) -> str:
        """Generate question when multiple values are possible."""
        options = "\n".join([f"{i+1}. {value}" for i, value in enumerate(values)])
        return f"I found multiple options for {entity_name}:\n{options}\nWhich one did you mean?"

class ClarificationManager:
    """Manages clarification requests and responses."""

    def __init__(self):
        self.pending_clarifications: Dict[str, Dict[str, Any]] = {}
        self.clarification_history: List[Dict[str, Any]] = []

    def request_clarification(
        self,
        session_id: str,
        clarification_type: str,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Request clarification from user."""
        self.pending_clarifications[session_id] = {
            'type': clarification_type,
            'question': question,
            'context': context,
            'timestamp': datetime.now()
        }
        return question

    def has_pending_clarification(self, session_id: str) -> bool:
        """Check if session has pending clarification."""
        return session_id in self.pending_clarifications

    def resolve_clarification(
        self,
        session_id: str,
        user_response: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve pending clarification with user response."""
        if session_id not in self.pending_clarifications:
            return None

        clarification = self.pending_clarifications[session_id]
        clarification_type = clarification['type']
        context = clarification['context']

        result = None

        if clarification_type == 'missing_entity':
            # Extract entity from response
            result = {
                'entity_name': context['entity_name'],
                'entity_value': user_response
            }

        elif clarification_type == 'ambiguous_intent':
            # Parse selection
            match = re.search(r'\d+', user_response)
            if match:
                selection = int(match.group()) - 1
                possible_intents = context['possible_intents']
                if 0 <= selection < len(possible_intents):
                    result = {
                        'intent': possible_intents[selection]
                    }

        elif clarification_type == 'confirmation':
            # Check for yes/no
            positive = any(word in user_response.lower() for word in ['yes', 'yeah', 'correct', 'right', 'yep'])
            negative = any(word in user_response.lower() for word in ['no', 'nope', 'wrong', 'incorrect'])

            if positive and not negative:
                result = {'confirmed': True}
            elif negative:
                result = {'confirmed': False}

        elif clarification_type == 'multiple_values':
            # Parse selection
            match = re.search(r'\d+', user_response)
            if match:
                selection = int(match.group()) - 1
                values = context['values']
                if 0 <= selection < len(values):
                    result = {
                        'entity_name': context['entity_name'],
                        'entity_value': values[selection]
                    }

        # Record history
        self.clarification_history.append({
            'session_id': session_id,
            'clarification': clarification,
            'user_response': user_response,
            'result': result,
            'timestamp': datetime.now()
        })

        # Clear pending
        del self.pending_clarifications[session_id]

        return result

    def get_clarification_count(self, session_id: str) -> int:
        """Get number of clarifications for session."""
        return sum(1 for h in self.clarification_history if h['session_id'] == session_id)

class CorrectionHandler:
    """Handles user corrections."""

    def __init__(self):
        self.correction_patterns = [
            (r"(?:no|nope),?\s*(?:i meant|i said|i mean)\s+(.+)", 1),
            (r"(?:actually|correction),?\s+(.+)", 1),
            (r"sorry,?\s+(.+)", 1),
            (r"(?:that's wrong|not right),?\s+(?:it's|its)\s+(.+)", 1)
        ]

    def is_correction(self, text: str) -> bool:
        """Check if text is a correction."""
        text_lower = text.lower()
        return any(
            re.search(pattern, text_lower)
            for pattern, _ in self.correction_patterns
        )

    def extract_correction(self, text: str) -> Optional[str]:
        """Extract the corrected information."""
        text_lower = text.lower()
        for pattern, group in self.correction_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(group).strip()
        return None

    def apply_correction(
        self,
        session: ConversationSession,
        correction: str
    ) -> bool:
        """Apply correction to the most recent turn."""
        if not session.turns:
            return False

        # Update the last turn
        last_turn = session.turns[-1]
        last_turn.metadata['corrected'] = True
        last_turn.metadata['original_message'] = last_turn.user_message
        last_turn.user_message = correction

        return True

# Example usage
def example_clarification_handling():
    """Demonstrate clarification and correction handling."""
    clarif_mgr = ClarificationManager()
    correction_handler = CorrectionHandler()

    session_id = "test_session"

    # Scenario 1: Missing entity
    question = clarif_mgr.request_clarification(
        session_id=session_id,
        clarification_type='missing_entity',
        question=ClarificationStrategy.missing_entity('city', ['Seattle', 'Portland']),
        context={'entity_name': 'city'}
    )
    print("Bot:", question)

    user_response = "Seattle"
    result = clarif_mgr.resolve_clarification(session_id, user_response)
    print(f"Resolved: {result}")

    # Scenario 2: Ambiguous intent
    question = clarif_mgr.request_clarification(
        session_id=session_id,
        clarification_type='ambiguous_intent',
        question=ClarificationStrategy.ambiguous_intent(['Check order status', 'Cancel order', 'Return order']),
        context={'possible_intents': ['order_status', 'cancel_order', 'return_order']}
    )
    print("\nBot:", question)

    user_response = "1"
    result = clarif_mgr.resolve_clarification(session_id, user_response)
    print(f"Resolved: {result}")

    # Scenario 3: Correction
    user_correction = "No, I meant Portland"
    if correction_handler.is_correction(user_correction):
        corrected = correction_handler.extract_correction(user_correction)
        print(f"\nDetected correction: {corrected}")

if __name__ == "__main__":
    example_clarification_handling()
```

## Personality and Consistency

Maintaining a consistent personality makes conversations more engaging and trustworthy.

### Personality Controller

```python
from typing import Dict, List, Optional
import random

class PersonalityTrait:
    """Defines a personality trait with examples."""

    def __init__(
        self,
        name: str,
        description: str,
        examples: List[str],
        response_templates: List[str]
    ):
        self.name = name
        self.description = description
        self.examples = examples
        self.response_templates = response_templates

class PersonalityProfile:
    """Complete personality profile for a conversational agent."""

    def __init__(
        self,
        name: str,
        description: str,
        traits: Dict[str, float],  # trait_name -> strength (0-1)
        tone: str,
        formality: float = 0.5,  # 0=casual, 1=formal
        emoji_usage: float = 0.3,
        humor: float = 0.3
    ):
        self.name = name
        self.description = description
        self.traits = traits
        self.tone = tone
        self.formality = formality
        self.emoji_usage = emoji_usage
        self.humor = humor

    def get_system_prompt(self) -> str:
        """Generate system prompt from personality."""
        prompt_parts = [
            f"You are {self.name}, a {self.description}.",
            f"Your tone is {self.tone}.",
        ]

        if self.formality < 0.3:
            prompt_parts.append("You communicate in a casual, friendly manner.")
        elif self.formality > 0.7:
            prompt_parts.append("You communicate in a professional, formal manner.")

        if self.emoji_usage > 0.5:
            prompt_parts.append("You occasionally use emojis to express emotions.")

        if self.humor > 0.5:
            prompt_parts.append("You have a good sense of humor and occasionally make lighthearted jokes.")

        # Add trait descriptions
        trait_descs = []
        for trait_name, strength in self.traits.items():
            if strength > 0.7:
                trait_descs.append(f"very {trait_name}")
            elif strength > 0.4:
                trait_descs.append(trait_name)

        if trait_descs:
            prompt_parts.append(f"You are {', '.join(trait_descs)}.")

        return " ".join(prompt_parts)

# Predefined personality profiles
PERSONALITY_PROFILES = {
    'helpful_assistant': PersonalityProfile(
        name="Helpful Assistant",
        description="friendly and knowledgeable AI assistant",
        traits={'helpful': 0.9, 'patient': 0.8, 'professional': 0.7},
        tone="warm and supportive",
        formality=0.6,
        emoji_usage=0.2,
        humor=0.3
    ),

    'friendly_companion': PersonalityProfile(
        name="Friendly Companion",
        description="casual and approachable friend",
        traits={'friendly': 0.9, 'empathetic': 0.8, 'encouraging': 0.8},
        tone="casual and warm",
        formality=0.2,
        emoji_usage=0.6,
        humor=0.6
    ),

    'professional_advisor': PersonalityProfile(
        name="Professional Advisor",
        description="expert professional advisor",
        traits={'professional': 0.9, 'knowledgeable': 0.9, 'precise': 0.8},
        tone="professional and authoritative",
        formality=0.9,
        emoji_usage=0.0,
        humor=0.1
    ),

    'enthusiastic_guide': PersonalityProfile(
        name="Enthusiastic Guide",
        description="energetic and passionate guide",
        traits={'enthusiastic': 0.9, 'encouraging': 0.9, 'creative': 0.7},
        tone="energetic and inspiring",
        formality=0.4,
        emoji_usage=0.7,
        humor=0.5
    )
}

class ResponseStyler:
    """Applies personality styling to responses."""

    def __init__(self, personality: PersonalityProfile):
        self.personality = personality

        # Greeting variations by formality
        self.greetings = {
            'high': ["Hello", "Good day", "Greetings"],
            'medium': ["Hi", "Hello there", "Hey there"],
            'low': ["Hey", "Hi there", "Yo", "What's up"]
        }

        # Acknowledgment variations
        self.acknowledgments = {
            'high': ["Certainly", "Of course", "Absolutely", "Indeed"],
            'medium': ["Sure", "Okay", "Alright", "Got it"],
            'low': ["Yeah", "Yep", "Cool", "Gotcha"]
        }

        # Emojis by context
        self.emojis = {
            'positive': ['😊', '👍', '✨', '🎉', '💪'],
            'thinking': ['🤔', '💭'],
            'understanding': ['👌', '✅', '💡'],
            'helpful': ['🔧', '📝', '🎯']
        }

    def style_response(self, response: str, context: str = 'neutral') -> str:
        """Apply personality styling to a response."""
        styled = response

        # Add emojis based on personality
        if random.random() < self.personality.emoji_usage:
            if context in self.emojis and self.emojis[context]:
                emoji = random.choice(self.emojis[context])
                styled = f"{styled} {emoji}"

        return styled

    def get_greeting(self) -> str:
        """Get a personality-appropriate greeting."""
        if self.personality.formality > 0.7:
            return random.choice(self.greetings['high'])
        elif self.personality.formality > 0.4:
            return random.choice(self.greetings['medium'])
        else:
            return random.choice(self.greetings['low'])

    def get_acknowledgment(self) -> str:
        """Get a personality-appropriate acknowledgment."""
        if self.personality.formality > 0.7:
            return random.choice(self.acknowledgments['high'])
        elif self.personality.formality > 0.4:
            return random.choice(self.acknowledgments['medium'])
        else:
            return random.choice(self.acknowledgments['low'])

class PersonalityController:
    """Controls personality consistency throughout conversation."""

    def __init__(self, profile: PersonalityProfile):
        self.profile = profile
        self.styler = ResponseStyler(profile)
        self.interaction_count = 0
        self.response_history: List[str] = []

    def get_system_prompt(self) -> str:
        """Get the system prompt with personality."""
        return self.profile.get_system_prompt()

    def process_response(self, response: str, context: str = 'neutral') -> str:
        """Process and style a response."""
        # Track response
        self.interaction_count += 1
        self.response_history.append(response)

        # Apply styling
        styled = self.styler.style_response(response, context)

        return styled

    def should_add_encouragement(self) -> bool:
        """Decide if encouragement should be added."""
        encouraging_trait = self.profile.traits.get('encouraging', 0)
        return random.random() < encouraging_trait

    def get_encouragement(self) -> str:
        """Get an encouraging phrase."""
        encouragements = [
            "You're doing great!",
            "Keep up the good work!",
            "I'm here to help!",
            "We'll figure this out together!",
            "Don't worry, we've got this!"
        ]
        return random.choice(encouragements)

# Example usage
def example_personality():
    """Demonstrate personality system."""
    # Create a friendly companion
    profile = PERSONALITY_PROFILES['friendly_companion']
    controller = PersonalityController(profile)

    print("System Prompt:")
    print(controller.get_system_prompt())
    print("\n" + "="*60 + "\n")

    # Simulate conversation
    responses = [
        ("Hello!", "positive"),
        ("I can help you with that.", "helpful"),
        ("Let me think about the best approach...", "thinking"),
        ("Got it! Here's what we'll do.", "understanding"),
        ("Great work!", "positive")
    ]

    for response, context in responses:
        styled = controller.process_response(response, context)
        print(f"Original: {response}")
        print(f"Styled:   {styled}")

        if controller.should_add_encouragement() and context == 'positive':
            print(f"Extra:    {controller.get_encouragement()}")
        print()

if __name__ == "__main__":
    example_personality()
```

## Memory Systems

Memory systems enable conversational AI to remember user preferences, past interactions, and important facts across sessions.

### Short-term and Long-term Memory

```python
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

@dataclass
class MemoryItem:
    """A single memory item."""
    key: str
    value: Any
    timestamp: datetime
    importance: float = 0.5  # 0-1, higher = more important
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def access(self):
        """Record an access to this memory."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def age_seconds(self) -> float:
        """Get age of memory in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()

    def relevance_score(self) -> float:
        """Calculate relevance score based on importance, recency, and access."""
        # Decay importance over time
        age_hours = self.age_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24)  # Decays over days

        # Boost from access count
        access_boost = min(0.3, self.access_count * 0.05)

        return self.importance * recency_factor + access_boost

class ShortTermMemory:
    """Working memory for current conversation."""

    def __init__(self, max_items: int = 50, ttl_seconds: int = 3600):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.memories: Dict[str, MemoryItem] = {}

    def store(self, key: str, value: Any, importance: float = 0.5):
        """Store an item in short-term memory."""
        memory = MemoryItem(
            key=key,
            value=value,
            timestamp=datetime.now(),
            importance=importance
        )
        self.memories[key] = memory

        # Prune if needed
        if len(self.memories) > self.max_items:
            self._prune()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from memory."""
        if key in self.memories:
            memory = self.memories[key]

            # Check TTL
            if memory.age_seconds() > self.ttl_seconds:
                del self.memories[key]
                return None

            memory.access()
            return memory.value
        return None

    def _prune(self):
        """Remove least relevant items."""
        # Sort by relevance score
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: x[1].relevance_score()
        )

        # Keep only top items
        keep_count = int(self.max_items * 0.8)
        to_keep = dict(sorted_memories[-keep_count:])
        self.memories = to_keep

    def clear(self):
        """Clear all short-term memory."""
        self.memories.clear()

    def get_all(self) -> Dict[str, Any]:
        """Get all items in memory."""
        return {k: v.value for k, v in self.memories.items()}

class LongTermMemory:
    """Persistent memory across sessions."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.memories: Dict[str, MemoryItem] = {}
        self.user_facts: Dict[str, List[str]] = defaultdict(list)
        self.preferences: Dict[str, Any] = {}

        if storage_path:
            self.load()

    def store(self, key: str, value: Any, importance: float = 0.7):
        """Store an item in long-term memory."""
        memory = MemoryItem(
            key=key,
            value=value,
            timestamp=datetime.now(),
            importance=importance
        )
        self.memories[key] = memory

        if self.storage_path:
            self.save()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from long-term memory."""
        if key in self.memories:
            memory = self.memories[key]
            memory.access()
            return memory.value
        return None

    def store_user_fact(self, user_id: str, fact: str):
        """Store a fact about the user."""
        self.user_facts[user_id].append(fact)
        if self.storage_path:
            self.save()

    def get_user_facts(self, user_id: str) -> List[str]:
        """Get all facts about a user."""
        return self.user_facts.get(user_id, [])

    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value
        if self.storage_path:
            self.save()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.preferences.get(key, default)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search memories by relevance."""
        # Simple keyword search + relevance scoring
        results = []
        query_lower = query.lower()

        for key, memory in self.memories.items():
            # Check if query matches key or value
            score = 0.0
            if query_lower in key.lower():
                score = memory.relevance_score() * 2.0
            elif query_lower in str(memory.value).lower():
                score = memory.relevance_score()

            if score > 0:
                results.append((key, memory.value, score))

        # Sort by score and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def save(self):
        """Save memories to disk."""
        if not self.storage_path:
            return

        data = {
            'memories': {
                k: {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'importance': v.importance,
                    'access_count': v.access_count,
                    'metadata': v.metadata
                }
                for k, v in self.memories.items()
            },
            'user_facts': dict(self.user_facts),
            'preferences': self.preferences
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load memories from disk."""
        if not self.storage_path or not Path(self.storage_path).exists():
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        # Load memories
        for key, mem_data in data.get('memories', {}).items():
            self.memories[key] = MemoryItem(
                key=key,
                value=mem_data['value'],
                timestamp=datetime.fromisoformat(mem_data['timestamp']),
                importance=mem_data['importance'],
                access_count=mem_data['access_count'],
                metadata=mem_data.get('metadata', {})
            )

        # Load user facts and preferences
        self.user_facts = defaultdict(list, data.get('user_facts', {}))
        self.preferences = data.get('preferences', {})

class MemoryManager:
    """Coordinates short-term and long-term memory."""

    def __init__(self, storage_path: Optional[str] = None):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory(storage_path)

    def remember(self, key: str, value: Any, persist: bool = False, importance: float = 0.5):
        """Store information in memory."""
        # Always store in short-term
        self.short_term.store(key, value, importance)

        # Store in long-term if important or requested
        if persist or importance > 0.7:
            self.long_term.store(key, value, importance)

    def recall(self, key: str) -> Optional[Any]:
        """Retrieve information from memory."""
        # Check short-term first (faster)
        value = self.short_term.retrieve(key)
        if value is not None:
            return value

        # Check long-term
        value = self.long_term.retrieve(key)
        if value is not None:
            # Promote to short-term for quick access
            self.short_term.store(key, value)
            return value

        return None

    def search_memory(self, query: str, top_k: int = 5) -> List[Tuple[str, Any, float]]:
        """Search across all memory."""
        return self.long_term.search(query, top_k)

    def remember_user_fact(self, user_id: str, fact: str):
        """Remember a fact about the user."""
        self.long_term.store_user_fact(user_id, fact)

    def get_user_context(self, user_id: str) -> str:
        """Get formatted user context from memory."""
        facts = self.long_term.get_user_facts(user_id)
        if not facts:
            return ""

        return "What I know about you:\n" + "\n".join(f"- {fact}" for fact in facts)

    def end_session(self):
        """End session and consolidate memories."""
        # Move important short-term memories to long-term
        for key, memory in self.short_term.memories.items():
            if memory.importance > 0.7 or memory.access_count > 3:
                self.long_term.store(key, memory.value, memory.importance)

        # Clear short-term
        self.short_term.clear()

# Example usage
def example_memory_systems():
    """Demonstrate memory systems."""
    memory = MemoryManager(storage_path="./conversation_memory.json")

    user_id = "user123"

    # Store some information
    memory.remember("user_name", "Alice", persist=True, importance=0.9)
    memory.remember("favorite_color", "blue", persist=True, importance=0.8)
    memory.remember("last_order", "12345", importance=0.6)

    # Remember user facts
    memory.remember_user_fact(user_id, "Likes spicy food")
    memory.remember_user_fact(user_id, "Prefers email notifications")
    memory.remember_user_fact(user_id, "Lives in Seattle")

    # Recall information
    print(f"User name: {memory.recall('user_name')}")
    print(f"Favorite color: {memory.recall('favorite_color')}")
    print(f"Last order: {memory.recall('last_order')}")

    # Get user context
    print("\n" + "="*60)
    print(memory.get_user_context(user_id))

    # Search memory
    print("\n" + "="*60)
    print("Search results for 'Seattle':")
    results = memory.search_memory("Seattle")
    for key, value, score in results:
        print(f"- {key}: {value} (score: {score:.2f})")

    # End session (consolidate memories)
    memory.end_session()

if __name__ == "__main__":
    example_memory_systems()
```

## Safety and Moderation

Safety systems protect users and maintain appropriate conversations.

### Safety Filter

```python
from typing import List, Dict, Optional, Tuple
import re

class ContentCategory(Enum):
    """Categories of potentially harmful content."""
    SAFE = "safe"
    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PERSONAL_INFO = "personal_info"
    SPAM = "spam"

@dataclass
class SafetyResult:
    """Result of safety check."""
    is_safe: bool
    categories: List[ContentCategory]
    confidence: float
    flagged_text: Optional[str] = None
    explanation: Optional[str] = None

class SafetyFilter:
    """Filters unsafe content from conversations."""

    def __init__(self):
        # Simple keyword-based filtering (production should use ML models)
        self.profanity_patterns = [
            r'\b(bad|word|here)\b',  # Replace with actual patterns
        ]

        self.hate_speech_patterns = [
            r'hate\s+group',
            r'discriminat(e|ion)',
        ]

        self.violence_patterns = [
            r'\b(hurt|harm|kill|attack)\b',
        ]

        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),  # SSN
            (r'\b\d{16}\b', 'Credit Card'),  # Credit card
            (r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 'Email'),  # Email
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'Phone'),  # Phone
        ]

        self.allowed_domains = {'example.com', 'trusted-site.com'}

    def check_content(self, text: str) -> SafetyResult:
        """Check if content is safe."""
        text_lower = text.lower()
        categories = []
        flagged_text = None

        # Check profanity
        for pattern in self.profanity_patterns:
            if re.search(pattern, text_lower):
                categories.append(ContentCategory.PROFANITY)
                break

        # Check hate speech
        for pattern in self.hate_speech_patterns:
            if re.search(pattern, text_lower):
                categories.append(ContentCategory.HATE_SPEECH)
                break

        # Check violence
        for pattern in self.violence_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Context-dependent - not all mentions are violent
                context_words = text_lower[max(0, match.start()-50):match.end()+50]
                if any(word in context_words for word in ['want to', 'will', 'going to']):
                    categories.append(ContentCategory.VIOLENCE)
                    flagged_text = match.group()
                break

        # Check for PII
        for pattern, pii_type in self.pii_patterns:
            if re.search(pattern, text):
                categories.append(ContentCategory.PERSONAL_INFO)
                flagged_text = pii_type
                break

        # Determine safety
        is_safe = len(categories) == 0
        confidence = 0.9 if is_safe else 0.8

        explanation = None
        if not is_safe:
            category_names = [c.value for c in categories]
            explanation = f"Content flagged for: {', '.join(category_names)}"

        return SafetyResult(
            is_safe=is_safe,
            categories=categories if not is_safe else [ContentCategory.SAFE],
            confidence=confidence,
            flagged_text=flagged_text,
            explanation=explanation
        )

    def sanitize_pii(self, text: str) -> str:
        """Remove or mask PII from text."""
        sanitized = text

        for pattern, pii_type in self.pii_patterns:
            sanitized = re.sub(pattern, f'[{pii_type} REDACTED]', sanitized)

        return sanitized

    def check_rate_limit(self, user_id: str, window_seconds: int = 60) -> bool:
        """Check if user is within rate limits."""
        # In production, use Redis or similar
        # Simplified version here
        return True  # Placeholder

class ModerationPolicy:
    """Defines moderation policies and responses."""

    def __init__(self):
        self.responses = {
            ContentCategory.PROFANITY: "I'm here to have a respectful conversation. Could we rephrase that?",
            ContentCategory.HATE_SPEECH: "I can't engage with that type of content. Let's keep our conversation respectful.",
            ContentCategory.VIOLENCE: "I'm not able to discuss violent topics. Is there something else I can help you with?",
            ContentCategory.SEXUAL_CONTENT: "I'm not designed to discuss adult content. How else can I assist you?",
            ContentCategory.SELF_HARM: "I'm concerned about what you've shared. Please reach out to a crisis helpline: 988 (US) or your local emergency services.",
            ContentCategory.ILLEGAL_ACTIVITY: "I can't provide assistance with illegal activities. Is there something legal I can help with?",
            ContentCategory.PERSONAL_INFO: "For your security, please don't share personal information like that. How else can I help?",
            ContentCategory.SPAM: "It looks like you're sending repetitive messages. How can I genuinely help you?"
        }

        self.escalation_threshold = 3
        self.user_violations: Dict[str, List[datetime]] = defaultdict(list)

    def get_response(self, category: ContentCategory) -> str:
        """Get appropriate response for a category."""
        return self.responses.get(category, "I'm not able to help with that request.")

    def record_violation(self, user_id: str, category: ContentCategory):
        """Record a policy violation."""
        self.user_violations[user_id].append(datetime.now())

    def should_escalate(self, user_id: str) -> bool:
        """Check if user should be escalated/blocked."""
        violations = self.user_violations.get(user_id, [])

        # Check violations in last hour
        recent_violations = [
            v for v in violations
            if (datetime.now() - v).total_seconds() < 3600
        ]

        return len(recent_violations) >= self.escalation_threshold

    def clear_violations(self, user_id: str, older_than_hours: int = 24):
        """Clear old violations."""
        if user_id in self.user_violations:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            self.user_violations[user_id] = [
                v for v in self.user_violations[user_id]
                if v > cutoff
            ]

class SafetyManager:
    """Manages all safety and moderation."""

    def __init__(self):
        self.filter = SafetyFilter()
        self.policy = ModerationPolicy()

    def check_message(self, user_id: str, message: str) -> Tuple[bool, Optional[str]]:
        """Check if message is safe and get response if not."""
        # Check content safety
        result = self.filter.check_content(message)

        if not result.is_safe:
            # Record violation
            for category in result.categories:
                if category != ContentCategory.SAFE:
                    self.policy.record_violation(user_id, category)

            # Check for escalation
            if self.policy.should_escalate(user_id):
                return False, "You've been temporarily blocked due to multiple policy violations. Please contact support if you believe this is an error."

            # Get appropriate response
            primary_category = result.categories[0]
            response = self.policy.get_response(primary_category)

            return False, response

        return True, None

    def sanitize_response(self, response: str) -> str:
        """Sanitize bot response before sending."""
        return self.filter.sanitize_pii(response)

# Example usage
def example_safety():
    """Demonstrate safety and moderation."""
    safety_mgr = SafetyManager()

    test_messages = [
        ("user1", "Hello! How are you?"),
        ("user2", "I want to hurt someone"),
        ("user3", "My SSN is 123-45-6789"),
        ("user4", "Can you help me with my homework?"),
    ]

    for user_id, message in test_messages:
        print(f"\nUser: {message}")
        is_safe, response = safety_mgr.check_message(user_id, message)

        if is_safe:
            print("✓ Message is safe")
        else:
            print(f"✗ Message flagged")
            print(f"Bot: {response}")

if __name__ == "__main__":
    example_safety()
```

## Multi-Turn Dialogue Patterns

Common patterns for handling multi-turn conversations.

### Dialogue Patterns

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

class DialoguePattern(ABC):
    """Base class for dialogue patterns."""

    @abstractmethod
    def is_applicable(self, session: ConversationSession) -> bool:
        """Check if pattern applies to current session."""
        pass

    @abstractmethod
    def execute(self, session: ConversationSession, user_input: str) -> str:
        """Execute the dialogue pattern."""
        pass

class FormFillingPattern(DialoguePattern):
    """Pattern for collecting multiple pieces of information."""

    def __init__(self, required_fields: List[str]):
        self.required_fields = required_fields
        self.field_prompts = {}
        self.collected_fields = {}

    def set_prompt(self, field: str, prompt: str):
        """Set the prompt for a field."""
        self.field_prompts[field] = prompt

    def is_applicable(self, session: ConversationSession) -> bool:
        """Check if form filling is needed."""
        return session.metadata.get('pattern') == 'form_filling'

    def execute(self, session: ConversationSession, user_input: str) -> str:
        """Execute form filling."""
        # Get current field being filled
        current_field = session.metadata.get('current_field')

        # Store the input for current field
        if current_field and user_input:
            self.collected_fields[current_field] = user_input

        # Find next empty field
        next_field = None
        for field in self.required_fields:
            if field not in self.collected_fields:
                next_field = field
                break

        # If all fields collected, complete the form
        if next_field is None:
            session.metadata['form_complete'] = True
            return f"Great! I have all the information: {self.collected_fields}"

        # Ask for next field
        session.metadata['current_field'] = next_field
        prompt = self.field_prompts.get(next_field, f"Please provide {next_field}:")
        return prompt

class ConfirmationPattern(DialoguePattern):
    """Pattern for confirming actions before execution."""

    def __init__(self):
        self.pending_action: Optional[Callable] = None
        self.action_description: Optional[str] = None

    def is_applicable(self, session: ConversationSession) -> bool:
        """Check if confirmation is needed."""
        return session.metadata.get('needs_confirmation', False)

    def set_action(self, action: Callable, description: str):
        """Set the action to be confirmed."""
        self.pending_action = action
        self.action_description = description

    def execute(self, session: ConversationSession, user_input: str) -> str:
        """Execute confirmation."""
        if not session.metadata.get('confirmation_asked', False):
            # First interaction - ask for confirmation
            session.metadata['confirmation_asked'] = True
            return f"Just to confirm: {self.action_description}. Is that correct? (yes/no)"

        # Parse response
        user_input_lower = user_input.lower()
        confirmed = any(word in user_input_lower for word in ['yes', 'yeah', 'yep', 'correct', 'right'])

        if confirmed and self.pending_action:
            # Execute the action
            result = self.pending_action()
            session.metadata['needs_confirmation'] = False
            session.metadata['confirmation_asked'] = False
            return f"Done! {result}"
        else:
            session.metadata['needs_confirmation'] = False
            session.metadata['confirmation_asked'] = False
            return "Okay, cancelled. How else can I help?"

class SlotFillingPattern(DialoguePattern):
    """Pattern for filling slots with flexible order."""

    def __init__(self):
        self.slots: Dict[str, Optional[str]] = {}
        self.slot_prompts: Dict[str, str] = {}
        self.slot_validators: Dict[str, Callable] = {}

    def add_slot(self, name: str, prompt: str, validator: Optional[Callable] = None):
        """Add a slot to fill."""
        self.slots[name] = None
        self.slot_prompts[name] = prompt
        if validator:
            self.slot_validators[name] = validator

    def is_applicable(self, session: ConversationSession) -> bool:
        """Check if slot filling is active."""
        return session.metadata.get('pattern') == 'slot_filling'

    def extract_slots(self, user_input: str, entities: Dict[str, Any]) -> Dict[str, str]:
        """Extract slot values from user input."""
        extracted = {}

        # Use detected entities
        for slot_name in self.slots:
            if slot_name in entities:
                extracted[slot_name] = entities[slot_name]

        return extracted

    def execute(self, session: ConversationSession, user_input: str) -> str:
        """Execute slot filling."""
        # Extract any slots from current input
        entities = session.turns[-1].entities if session.turns else {}
        extracted = self.extract_slots(user_input, entities)

        # Validate and fill slots
        for slot_name, value in extracted.items():
            validator = self.slot_validators.get(slot_name)
            if validator:
                if validator(value):
                    self.slots[slot_name] = value
                else:
                    return f"Invalid {slot_name}. {self.slot_prompts[slot_name]}"
            else:
                self.slots[slot_name] = value

        # Check if all slots filled
        empty_slots = [name for name, value in self.slots.items() if value is None]

        if not empty_slots:
            session.metadata['slots_complete'] = True
            return f"Perfect! I have all the details: {self.slots}"

        # Ask for next empty slot
        next_slot = empty_slots[0]
        return self.slot_prompts[next_slot]

class GuidedNavigationPattern(DialoguePattern):
    """Pattern for guiding users through options."""

    def __init__(self):
        self.navigation_tree = {}
        self.current_path = []

    def set_tree(self, tree: Dict[str, Any]):
        """Set the navigation tree."""
        self.navigation_tree = tree

    def is_applicable(self, session: ConversationSession) -> bool:
        """Check if guided navigation is active."""
        return session.metadata.get('pattern') == 'guided_navigation'

    def execute(self, session: ConversationSession, user_input: str) -> str:
        """Execute guided navigation."""
        # Get current node
        current_node = self.navigation_tree
        for path_item in self.current_path:
            current_node = current_node.get(path_item, {})

        # If user made a selection
        if user_input.isdigit():
            selection = int(user_input) - 1
            options = list(current_node.keys())

            if 0 <= selection < len(options):
                selected = options[selection]
                self.current_path.append(selected)

                next_node = current_node[selected]

                # If leaf node, return result
                if isinstance(next_node, str):
                    result = next_node
                    self.current_path = []  # Reset
                    return result

                # Otherwise show next options
                return self._format_options(next_node)

        # Show current options
        return self._format_options(current_node)

    def _format_options(self, node: Dict[str, Any]) -> str:
        """Format options for display."""
        if not node:
            return "No options available."

        lines = ["Please select an option:"]
        for i, option in enumerate(node.keys(), 1):
            lines.append(f"{i}. {option}")

        return "\n".join(lines)

# Combining patterns
class DialogueManager:
    """Manages multiple dialogue patterns."""

    def __init__(self):
        self.patterns: List[DialoguePattern] = []
        self.active_pattern: Optional[DialoguePattern] = None

    def add_pattern(self, pattern: DialoguePattern):
        """Add a dialogue pattern."""
        self.patterns.append(pattern)

    def process_turn(self, session: ConversationSession, user_input: str) -> str:
        """Process a turn using appropriate pattern."""
        # Check if we have an active pattern
        if self.active_pattern and self.active_pattern.is_applicable(session):
            return self.active_pattern.execute(session, user_input)

        # Find applicable pattern
        for pattern in self.patterns:
            if pattern.is_applicable(session):
                self.active_pattern = pattern
                return pattern.execute(session, user_input)

        # No pattern applicable
        return "How can I help you?"

# Example usage
def example_dialogue_patterns():
    """Demonstrate dialogue patterns."""
    # Create a form filling pattern
    form_pattern = FormFillingPattern(['name', 'email', 'phone'])
    form_pattern.set_prompt('name', "What's your name?")
    form_pattern.set_prompt('email', "What's your email address?")
    form_pattern.set_prompt('phone', "What's your phone number?")

    # Create session
    manager = ConversationStateManager()
    session = manager.create_session("user123")
    session.metadata['pattern'] = 'form_filling'

    # Simulate form filling
    print("Starting form filling...\n")

    response = form_pattern.execute(session, "")
    print(f"Bot: {response}")

    response = form_pattern.execute(session, "John Doe")
    print(f"User: John Doe")
    print(f"Bot: {response}")

    response = form_pattern.execute(session, "john@example.com")
    print(f"User: john@example.com")
    print(f"Bot: {response}")

    response = form_pattern.execute(session, "555-1234")
    print(f"User: 555-1234")
    print(f"Bot: {response}")

if __name__ == "__main__":
    example_dialogue_patterns()
```

## Production Deployment

Production-ready conversational AI requires robust infrastructure, monitoring, and optimization.

### Production Chatbot

```python
from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatbot:
    """Production-ready conversational AI system."""

    def __init__(
        self,
        personality_profile: PersonalityProfile,
        memory_storage_path: Optional[str] = None,
        enable_safety: bool = True
    ):
        # Core components
        self.state_manager = ConversationStateManager()
        self.context_tracker = ContextTracker()
        self.personality_controller = PersonalityController(personality_profile)
        self.memory_manager = MemoryManager(memory_storage_path)
        self.clarification_manager = ClarificationManager()
        self.correction_handler = CorrectionHandler()
        self.dialogue_manager = DialogueManager()

        # Safety components
        self.enable_safety = enable_safety
        if enable_safety:
            self.safety_manager = SafetyManager()

        # Metrics
        self.metrics = {
            'total_conversations': 0,
            'total_turns': 0,
            'avg_conversation_length': 0.0,
            'clarification_rate': 0.0,
            'safety_flags': 0
        }

    async def start_conversation(self, user_id: str) -> Dict[str, Any]:
        """Start a new conversation."""
        session = self.state_manager.create_session(user_id)
        session.state = ConversationState.GREETING

        # Get user context from memory
        user_context = self.memory_manager.get_user_context(user_id)

        # Generate greeting
        greeting = self.personality_controller.styler.get_greeting()

        if user_context:
            response = f"{greeting}! Nice to see you again. {user_context}"
        else:
            response = f"{greeting}! How can I help you today?"

        response = self.personality_controller.process_response(response, 'positive')

        # Add first turn
        self.state_manager.add_turn(
            session,
            user_message="[Conversation started]",
            bot_response=response,
            intent="greeting"
        )

        self.metrics['total_conversations'] += 1

        return {
            'session_id': session.session_id,
            'response': response,
            'state': session.state.value
        }

    async def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict[str, Any]:
        """Process a user message."""
        session = self.state_manager.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}

        try:
            # Safety check
            if self.enable_safety:
                is_safe, safety_response = self.safety_manager.check_message(
                    session.user_id,
                    user_message
                )
                if not is_safe:
                    self.metrics['safety_flags'] += 1
                    return {
                        'session_id': session_id,
                        'response': safety_response,
                        'state': session.state.value,
                        'flagged': True
                    }

            # Check for pending clarification
            if self.clarification_manager.has_pending_clarification(session_id):
                result = self.clarification_manager.resolve_clarification(
                    session_id,
                    user_message
                )

                if result:
                    # Clarification resolved, continue processing
                    self.state_manager.set_metadata(session, 'clarification_provided', True)
                else:
                    # Need more clarification
                    return {
                        'session_id': session_id,
                        'response': "I'm still not sure. Could you clarify?",
                        'state': session.state.value
                    }

            # Check for correction
            if self.correction_handler.is_correction(user_message):
                correction = self.correction_handler.extract_correction(user_message)
                if correction:
                    self.correction_handler.apply_correction(session, correction)
                    user_message = correction

            # Resolve references in user input
            resolved_message = self.context_tracker.resolve_user_input(user_message)

            # Generate response (simplified - in production use LLM)
            response = await self._generate_response(session, resolved_message)

            # Apply personality styling
            response = self.personality_controller.process_response(response)

            # Safety check on response
            if self.enable_safety:
                response = self.safety_manager.sanitize_response(response)

            # Create turn
            turn = self.state_manager.add_turn(
                session,
                user_message=user_message,
                bot_response=response,
                intent=self._detect_intent(resolved_message),  # Simplified
                entities=self._extract_entities(resolved_message)  # Simplified
            )

            # Update context
            self.context_tracker.add_turn(turn)

            # Update state
            self.state_manager.update_state(session)

            # Update metrics
            self.metrics['total_turns'] += 1
            self._update_metrics()

            return {
                'session_id': session_id,
                'response': response,
                'state': session.state.value,
                'turn_id': turn.turn_id
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                'session_id': session_id,
                'response': "I'm sorry, I encountered an error. Could you try rephrasing that?",
                'state': session.state.value,
                'error': str(e)
            }

    async def _generate_response(self, session: ConversationSession, message: str) -> str:
        """Generate response using LLM (simplified version)."""
        # In production, this would call an LLM API with proper context

        # Get context for LLM
        context = self.context_tracker.get_context_for_llm()

        # Get personality prompt
        system_prompt = self.personality_controller.get_system_prompt()

        # Simplified response generation (replace with actual LLM call)
        response = f"I understand you said: {message}. How can I help with that?"

        return response

    def _detect_intent(self, message: str) -> str:
        """Detect user intent (simplified)."""
        # In production, use proper intent classification
        message_lower = message.lower()

        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            return 'greeting'
        elif any(word in message_lower for word in ['bye', 'goodbye']):
            return 'farewell'
        elif '?' in message:
            return 'question'
        else:
            return 'statement'

    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message (simplified)."""
        # In production, use proper NER
        entities = {}

        # Simple email detection
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', message)
        if email_match:
            entities['email'] = email_match.group()

        return entities

    def _update_metrics(self):
        """Update conversation metrics."""
        if self.metrics['total_conversations'] > 0:
            self.metrics['avg_conversation_length'] = (
                self.metrics['total_turns'] / self.metrics['total_conversations']
            )

        clarifications = len(self.clarification_manager.clarification_history)
        if self.metrics['total_turns'] > 0:
            self.metrics['clarification_rate'] = clarifications / self.metrics['total_turns']

    async def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """End a conversation."""
        session = self.state_manager.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}

        # End session
        self.state_manager.end_session(session)

        # Consolidate memories
        self.memory_manager.end_session()

        # Log metrics
        logger.info(f"Conversation ended: {session_id}, turns: {session.turn_count}, duration: {session.duration_seconds}s")

        return {
            'session_id': session_id,
            'turn_count': session.turn_count,
            'duration_seconds': session.duration_seconds,
            'status': 'ended'
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get chatbot metrics."""
        return self.metrics.copy()

# Example usage
async def example_production_chatbot():
    """Demonstrate production chatbot."""
    # Create chatbot with friendly personality
    chatbot = ProductionChatbot(
        personality_profile=PERSONALITY_PROFILES['friendly_companion'],
        memory_storage_path="./chatbot_memory.json",
        enable_safety=True
    )

    # Start conversation
    result = await chatbot.start_conversation("user123")
    print(f"Bot: {result['response']}\n")

    # Simulate conversation
    messages = [
        "Hi! I need help with my order",
        "Order number 12345",
        "I want to change the shipping address",
        "123 Main St, Seattle, WA",
        "Thanks!"
    ]

    for message in messages:
        print(f"User: {message}")
        result = await chatbot.process_message(result['session_id'], message)
        print(f"Bot: {result['response']}\n")

    # End conversation
    end_result = await chatbot.end_conversation(result['session_id'])
    print(f"Conversation ended: {end_result['turn_count']} turns, {end_result['duration_seconds']:.1f}s")

    # Show metrics
    print("\nMetrics:")
    for key, value in chatbot.get_metrics().items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    asyncio.run(example_production_chatbot())
```

## Summary

**Conversational AI** enables natural, multi-turn dialogue between humans and machines through sophisticated state management, context tracking, and personality control:

**Core Components**:

- **State Management**: Track conversation flow and transitions
- **Context Tracking**: Maintain history and resolve references
- **Clarifications**: Handle unclear inputs and corrections
- **Personality**: Consistent character and tone
- **Memory**: Remember user preferences and facts
- **Safety**: Filter harmful content and moderate conversations
- **Patterns**: Reusable dialogue structures
- **Production**: Robust deployment with monitoring

**Key Techniques**:

- Conversation state machines with transitions
- Context windows and entity tracking
- Reference resolution for pronouns
- Form filling and slot filling patterns
- Short-term and long-term memory systems
- Content safety filtering
- Personality profiles and styling
- Rate limiting and policy enforcement

**Best Practices**:

- Design clear state transitions
- Maintain conversation context efficiently
- Handle ambiguity with clarifications
- Implement consistent personality
- Store important information in memory
- Enforce safety policies
- Use appropriate dialogue patterns
- Monitor metrics and performance
- Provide graceful error handling
- Enable easy session recovery

**Production Considerations**:

- Use persistent storage for sessions
- Implement rate limiting
- Add comprehensive logging
- Monitor conversation metrics
- Handle concurrent conversations
- Optimize context window size
- Cache memory lookups
- Deploy with high availability
- Test edge cases thoroughly
- Plan for scalability

Conversational AI transforms human-computer interaction by enabling natural, context-aware dialogue that remembers past interactions and adapts to user preferences.

## Next Steps

- Apply **[Question Answering](question-answering.md)** for information retrieval in conversations
- Use **[Text Generation](text-generation.md)** for creative response generation
- Implement **[Semantic Search](semantic-search.md)** for knowledge retrieval
- Explore **[Summarization](summarization.md)** for conversation summaries
- Study **[Sentiment Analysis](sentiment-analysis.md)** for emotion detection
- Learn **[Named Entity Recognition](named-entity-recognition.md)** for better entity tracking
- Implement **[Intent Classification](intent-classification.md)** for understanding user goals
- Use **[Text Classification](text-classification.md)** for topic detection
