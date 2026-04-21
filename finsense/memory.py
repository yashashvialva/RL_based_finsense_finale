import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple


class MemorySystem:
    """Persistent memory system using SQLite to store and retrieve decision patterns."""

    def __init__(self, db_path: str = "finsense_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    episode INTEGER,
                    day INTEGER,
                    expense_type TEXT,
                    necessity TEXT,
                    context TEXT,
                    event_type TEXT,
                    price_multiplier REAL,
                    balance REAL,
                    days_left INTEGER,
                    action TEXT,
                    reward REAL,
                    outcome TEXT
                )
            ''')
            # Add index for faster retrieval queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_decisions_lookup
                ON decisions (expense_type, context, event_type)
            ''')
            conn.commit()

    def store_decision(self, episode: int, day: int, expense_type: str, necessity: str,
                      context: str, event_type: str, price_multiplier: float,
                      balance: float, days_left: int, action: str, reward: float,
                      outcome: str):
        """Store a decision in memory."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO decisions
                (episode, day, expense_type, necessity, context, event_type,
                 price_multiplier, balance, days_left, action, reward, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (episode, day, expense_type, necessity, context, event_type,
                  price_multiplier, balance, days_left, action, reward, outcome))
            conn.commit()

    def retrieve_similar_cases(self, expense_type: str, context: str, event_type: str,
                              necessity: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top similar past cases with fallback matching.
        
        Tries exact match first, then progressively relaxes criteria:
        1. expense_type + context + event_type (exact)
        2. expense_type + context (drop event)
        3. expense_type + necessity (drop context and event)
        4. expense_type only (broadest)
        """
        with sqlite3.connect(self.db_path) as conn:
            # Level 1: Exact match on all three keys
            cursor = conn.execute('''
                SELECT action, reward, outcome, balance, days_left, necessity, context, event_type
                FROM decisions
                WHERE expense_type = ? AND context = ? AND event_type = ?
                ORDER BY reward DESC
                LIMIT ?
            ''', (expense_type, context, event_type, limit))
            results = cursor.fetchall()

            # Level 2: Fallback - expense_type + context
            if len(results) < 3:
                cursor = conn.execute('''
                    SELECT action, reward, outcome, balance, days_left, necessity, context, event_type
                    FROM decisions
                    WHERE expense_type = ? AND context = ?
                    ORDER BY reward DESC
                    LIMIT ?
                ''', (expense_type, context, limit))
                results = cursor.fetchall()

            # Level 3: Fallback - expense_type + necessity
            if len(results) < 3 and necessity:
                cursor = conn.execute('''
                    SELECT action, reward, outcome, balance, days_left, necessity, context, event_type
                    FROM decisions
                    WHERE expense_type = ? AND necessity = ?
                    ORDER BY reward DESC
                    LIMIT ?
                ''', (expense_type, necessity, limit))
                results = cursor.fetchall()

            # Level 4: Broadest - just expense_type
            if len(results) < 3:
                cursor = conn.execute('''
                    SELECT action, reward, outcome, balance, days_left, necessity, context, event_type
                    FROM decisions
                    WHERE expense_type = ?
                    ORDER BY reward DESC
                    LIMIT ?
                ''', (expense_type, limit))
                results = cursor.fetchall()

            return [
                {
                    'action': row[0],
                    'reward': row[1],
                    'outcome': row[2],
                    'balance': row[3],
                    'days_left': row[4],
                    'necessity': row[5],
                    'context': row[6],
                    'event_type': row[7],
                }
                for row in results
            ]

    def get_memory_bias(self, expense_type: str, context: str, event_type: str,
                       necessity: str = None) -> Optional[str]:
        """Get the most successful action from memory for similar situations.
        
        Returns the action with highest confidence based on past successful outcomes.
        Only returns a bias when there's sufficient evidence (>= 3 matching cases).
        """
        similar_cases = self.retrieve_similar_cases(
            expense_type, context, event_type, necessity=necessity, limit=15
        )

        if len(similar_cases) < 2:
            return None

        # Weight actions by reward (higher reward = more weight)
        action_scores: Dict[str, float] = {}
        action_counts: Dict[str, int] = {}

        for case in similar_cases:
            action = case['action']
            reward = case['reward']

            if action not in action_scores:
                action_scores[action] = 0.0
                action_counts[action] = 0

            action_scores[action] += reward
            action_counts[action] += 1

        if not action_scores:
            return None

        # Calculate average reward per action
        action_avg = {
            action: action_scores[action] / action_counts[action]
            for action in action_scores
        }

        # Return the action with highest average reward
        best_action = max(action_avg, key=action_avg.get)
        best_count = action_counts[best_action]

        # Only return bias if we have >= 3 cases supporting it
        if best_count >= 3:
            return best_action

        return None

    def get_memory_bias_with_confidence(self, expense_type: str, context: str,
                                        event_type: str, necessity: str = None) -> Tuple[Optional[str], float]:
        """Get memory bias along with a confidence score (0.0 to 1.0).
        
        Returns (action, confidence) where confidence is based on:
        - number of matching cases
        - consistency of recommended action
        - average reward of recommended action
        """
        similar_cases = self.retrieve_similar_cases(
            expense_type, context, event_type, necessity=necessity, limit=15
        )

        if len(similar_cases) < 2:
            return None, 0.0

        action_scores: Dict[str, float] = {}
        action_counts: Dict[str, int] = {}

        for case in similar_cases:
            action = case['action']
            reward = case['reward']

            if action not in action_scores:
                action_scores[action] = 0.0
                action_counts[action] = 0

            # Store the raw reward as adjusted by env.py
            action_scores[action] += reward
            action_counts[action] += 1

        if not action_scores:
            return None, 0.0

        # Find best action
        action_avg = {
            action: action_scores[action] / action_counts[action]
            for action in action_scores
        }
        best_action = max(action_avg, key=action_avg.get)
        best_count = action_counts[best_action]
        best_avg_reward = action_avg[best_action]
        total_cases = len(similar_cases)

        # Confidence factors:
        # 1. Consistency: what fraction of cases agree on the best action
        consistency = best_count / total_cases

        # 2. Volume: more cases = more confidence (saturates at 8 to be aggressive)
        volume = min(1.0, best_count / 8.0)

        # 3. Reward quality: higher average reward = more confidence
        reward_quality = min(1.0, max(0.0, best_avg_reward))

        # We increase consistency weight because if an action consistently led to success, we want it
        confidence = (consistency * 0.5) + (volume * 0.2) + (reward_quality * 0.3)

        return best_action, confidence

    def clear_memory(self):
        """Clear all stored memories."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM decisions')
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM decisions')
            total_decisions = cursor.fetchone()[0]

            cursor = conn.execute('SELECT AVG(reward) FROM decisions')
            avg_reward = cursor.fetchone()[0] or 0.0

            cursor = conn.execute('SELECT COUNT(DISTINCT episode) FROM decisions')
            total_episodes = cursor.fetchone()[0]

            cursor = conn.execute('''
                SELECT action, COUNT(*) as cnt, AVG(reward) as avg_r
                FROM decisions
                GROUP BY action
                ORDER BY avg_r DESC
            ''')
            action_stats = [
                {'action': row[0], 'count': row[1], 'avg_reward': round(row[2], 4)}
                for row in cursor.fetchall()
            ]

            return {
                'total_decisions': total_decisions,
                'total_episodes': total_episodes,
                'average_reward': round(avg_reward, 4),
                'action_stats': action_stats,
            }