import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

@dataclass
class Theory:
    """Represents a validated theory about learning"""
    statement: str
    evidence: Dict[str, Any]
    confidence: float
    domain: str
    date_discovered: datetime
    validation_count: int = 0
    counter_examples: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.counter_examples is None:
            self.counter_examples = []

@dataclass
class Principle:
    """Represents an extracted general principle"""
    rule: str
    conditions: Dict[str, Any]
    effectiveness: float
    domain: str
    date_extracted: datetime
    supporting_experiments: Optional[List[str]] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.supporting_experiments is None:
            self.supporting_experiments = []

class TheoryBuilder:
    """Build theories about learning"""
    
    def build_theory(self, observations, experiments):
        """Construct theory from evidence"""
        
        # Identify common patterns
        patterns = self._find_patterns(observations)
        
        # Formulate rules
        rules = []
        for pattern in patterns:
            rule = self._formulate_rule(pattern)
            supporting_evidence = self._gather_evidence(rule, experiments)
            
            if len(supporting_evidence) >= 3:  # Require multiple confirmations
                rules.append({
                    'rule': rule,
                    'evidence': supporting_evidence,
                    'confidence': self._calculate_confidence(supporting_evidence)
                })
        
        # Combine into theory
        theory = {
            'name': self._name_theory(rules),
            'rules': rules,
            'domain': self._identify_domain(observations),
            'predictive_power': self._test_predictions(rules),
            'falsifiable': self._check_falsifiability(rules)
        }
        
        return theory

    def _find_patterns(self, observations):
        """Pattern recognition: identify common patterns in observations"""
        if not observations:
            return []
        
        patterns = []
        # Assume observations are dicts
        all_keys = set()
        for obs in observations:
            if isinstance(obs, dict):
                all_keys.update(obs.keys())
        
        for key in all_keys:
            values = [obs.get(key) for obs in observations if isinstance(obs, dict) and key in obs]
            if len(values) > len(observations) * 0.5:  # At least half have this key
                # Check if values are similar
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    try:
                        mean_val = np.mean([v for v in values if v is not None])
                        std_val = np.std([v for v in values if v is not None])
                        if std_val / (abs(mean_val) + 1e-6) < 0.3:  # Low coefficient of variation
                            patterns.append({'type': 'numeric', 'key': key, 'value': mean_val})
                    except:
                        pass
                elif len(set(str(v) for v in values if v is not None)) == 1:
                    patterns.append({'type': 'categorical', 'key': key, 'value': values[0]})
        
        return patterns

    def _formulate_rule(self, pattern):
        """Rule induction: formulate a rule from a pattern"""
        if pattern['type'] == 'numeric':
            return f"{pattern['key']} is approximately {pattern['value']:.2f}"
        elif pattern['type'] == 'categorical':
            return f"{pattern['key']} is {pattern['value']}"
        return "Unknown pattern"

    def _gather_evidence(self, rule, experiments):
        """Gather supporting evidence from experiments"""
        evidence = []
        for exp in experiments:
            if isinstance(exp, dict):
                # Check if experiment supports the rule
                if 'is approximately' in rule:
                    parts = rule.split(' is approximately ')
                    if len(parts) == 2:
                        key, value_str = parts
                        key = key.strip()
                        try:
                            value = float(value_str)
                            exp_value = exp.get(key)
                            if exp_value is not None and abs(float(exp_value) - value) < 0.1 * abs(value):
                                evidence.append(exp)
                        except:
                            pass
                elif 'is ' in rule:
                    parts = rule.split(' is ')
                    if len(parts) == 2:
                        key, value = parts
                        key = key.strip()
                        value = value.strip()
                        if exp.get(key) == value:
                            evidence.append(exp)
        return evidence

    def _calculate_confidence(self, supporting_evidence):
        """Calculate confidence based on evidence strength"""
        if not supporting_evidence:
            return 0.0
        # Confidence based on number and consistency of evidence
        num_evidence = len(supporting_evidence)
        # Simple formula: more evidence = higher confidence, max 1.0
        confidence = min(1.0, num_evidence / 10.0)
        return confidence

    def _name_theory(self, rules):
        """Name the theory based on its rules"""
        if not rules:
            return "Empty Theory"
        # Name based on first rule
        first_rule = rules[0]['rule']
        return f"Theory of {first_rule[:50]}{'...' if len(first_rule) > 50 else ''}"

    def _identify_domain(self, observations):
        """Identify the domain from observations"""
        domains = []
        for obs in observations:
            if isinstance(obs, dict) and 'domain' in obs:
                domains.append(obs['domain'])
        if domains:
            from collections import Counter
            return Counter(domains).most_common(1)[0][0]
        return 'general'

    def _test_predictions(self, rules):
        """Test predictive power of rules"""
        if not rules:
            return 0.0
        # Simple: average confidence as proxy for predictive power
        confidences = [r['confidence'] for r in rules]
        return np.mean(confidences) if confidences else 0.0

    def _check_falsifiability(self, rules):
        """Check if theory is falsifiable"""
        if not rules:
            return False
        # Theory is falsifiable if rules make specific claims that can be tested
        return all('is' in r['rule'] for r in rules)

class KnowledgeBase:
    """Persistent learning about learning"""

    def __init__(self, storage_path: str = "knowledge_base.db", use_sqlite: bool = True):
        self.theories: List[Theory] = []
        self.principles: List[Principle] = []
        self.strategies: List[Dict[str, Any]] = []
        self.counter_examples: List[Dict[str, Any]] = []

        self.storage_path = storage_path
        self.use_sqlite = use_sqlite

        if use_sqlite:
            self._init_sqlite()
        else:
            self._load_json()

    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(self.storage_path)
        self.conn.execute('PRAGMA foreign_keys = ON')

        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS theories (
                id INTEGER PRIMARY KEY,
                statement TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT NOT NULL,
                date_discovered TEXT NOT NULL,
                validation_count INTEGER DEFAULT 0,
                counter_examples TEXT DEFAULT '[]'
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS principles (
                id INTEGER PRIMARY KEY,
                rule TEXT NOT NULL,
                conditions TEXT NOT NULL,
                effectiveness REAL NOT NULL,
                domain TEXT NOT NULL,
                date_extracted TEXT NOT NULL,
                supporting_experiments TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.0
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                parameters TEXT NOT NULL,
                performance REAL NOT NULL,
                domain TEXT NOT NULL,
                date_created TEXT NOT NULL
            )
        ''')

        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS counter_examples (
                id INTEGER PRIMARY KEY,
                theory_id INTEGER,
                description TEXT NOT NULL,
                evidence TEXT NOT NULL,
                date_found TEXT NOT NULL,
                FOREIGN KEY (theory_id) REFERENCES theories (id)
            )
        ''')

        self.conn.commit()
        self._load_from_sqlite()

    def _load_from_sqlite(self):
        """Load data from SQLite database"""
        # Load theories
        cursor = self.conn.execute('SELECT * FROM theories')
        for row in cursor:
            evidence = json.loads(row[2])
            counter_examples = json.loads(row[7])
            theory = Theory(
                statement=row[1],
                evidence=evidence,
                confidence=row[3],
                domain=row[4],
                date_discovered=datetime.fromisoformat(row[5]),
                validation_count=row[6],
                counter_examples=counter_examples
            )
            self.theories.append(theory)

        # Load principles
        cursor = self.conn.execute('SELECT * FROM principles')
        for row in cursor:
            conditions = json.loads(row[2])
            supporting_experiments = json.loads(row[6])
            principle = Principle(
                rule=row[1],
                conditions=conditions,
                effectiveness=row[3],
                domain=row[4],
                date_extracted=datetime.fromisoformat(row[5]),
                supporting_experiments=supporting_experiments,
                confidence=row[7]
            )
            self.principles.append(principle)

        # Load strategies
        cursor = self.conn.execute('SELECT * FROM strategies')
        for row in cursor:
            parameters = json.loads(row[3])
            strategy = {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'parameters': parameters,
                'performance': row[4],
                'domain': row[5],
                'date_created': datetime.fromisoformat(row[6])
            }
            self.strategies.append(strategy)

        # Load counter examples
        cursor = self.conn.execute('SELECT * FROM counter_examples')
        for row in cursor:
            counter_example = {
                'id': row[0],
                'theory_id': row[1],
                'description': row[2],
                'evidence': json.loads(row[3]),
                'date_found': datetime.fromisoformat(row[4])
            }
            self.counter_examples.append(counter_example)

    def _load_json(self):
        """Load data from JSON file"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.theories = [Theory(**t) for t in data.get('theories', [])]
                self.principles = [Principle(**p) for p in data.get('principles', [])]
                self.strategies = data.get('strategies', [])
                self.counter_examples = data.get('counter_examples', [])

    def _save_sqlite(self):
        """Save data to SQLite database"""
        # Clear existing data
        self.conn.execute('DELETE FROM theories')
        self.conn.execute('DELETE FROM principles')
        self.conn.execute('DELETE FROM strategies')
        self.conn.execute('DELETE FROM counter_examples')

        # Save theories
        for theory in self.theories:
            self.conn.execute('''
                INSERT INTO theories (statement, evidence, confidence, domain, date_discovered, validation_count, counter_examples)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                theory.statement,
                json.dumps(theory.evidence),
                theory.confidence,
                theory.domain,
                theory.date_discovered.isoformat(),
                theory.validation_count,
                json.dumps(theory.counter_examples)
            ))

        # Save principles
        for principle in self.principles:
            self.conn.execute('''
                INSERT INTO principles (rule, conditions, effectiveness, domain, date_extracted, supporting_experiments, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                principle.rule,
                json.dumps(principle.conditions),
                principle.effectiveness,
                principle.domain,
                principle.date_extracted.isoformat(),
                json.dumps(principle.supporting_experiments),
                principle.confidence
            ))

        # Save strategies
        for strategy in self.strategies:
            self.conn.execute('''
                INSERT INTO strategies (name, description, parameters, performance, domain, date_created)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                strategy['name'],
                strategy['description'],
                json.dumps(strategy['parameters']),
                strategy['performance'],
                strategy['domain'],
                strategy['date_created'].isoformat()
            ))

        # Save counter examples
        for counter_example in self.counter_examples:
            self.conn.execute('''
                INSERT INTO counter_examples (theory_id, description, evidence, date_found)
                VALUES (?, ?, ?, ?)
            ''', (
                counter_example['theory_id'],
                counter_example['description'],
                json.dumps(counter_example['evidence']),
                counter_example['date_found'].isoformat()
            ))

        self.conn.commit()

    def _save_json(self):
        """Save data to JSON file"""
        data = {
            'theories': [asdict(t) for t in self.theories],
            'principles': [asdict(p) for p in self.principles],
            'strategies': self.strategies,
            'counter_examples': self.counter_examples
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def save(self):
        """Save knowledge base to storage"""
        if self.use_sqlite:
            self._save_sqlite()
        else:
            self._save_json()

    def add_theory(self, theory: Dict[str, Any]):
        """Add validated theory"""
        # Validate theory structure
        if not self._validate_theory(theory):
            raise ValueError("Invalid theory structure")

        theory_obj = Theory(
            statement=theory['statement'],
            evidence=theory['evidence'],
            confidence=theory['confidence'],
            domain=theory['domain'],
            date_discovered=datetime.now()
        )

        self.theories.append(theory_obj)
        self.save()
        return theory_obj

    def _validate_theory(self, theory: Dict[str, Any]) -> bool:
        """Validate theory structure and evidence"""
        required_fields = ['statement', 'evidence', 'confidence', 'domain']
        if not all(field in theory for field in required_fields):
            return False

        # Check confidence is reasonable
        if not (0.0 <= theory['confidence'] <= 1.0):
            return False

        # Check evidence has some substance
        if not theory['evidence']:
            return False

        return True

    def extract_principle(self, experiment_results: List[Dict[str, Any]]) -> Optional[Principle]:
        """Extract general principle from experiments"""
        if not experiment_results:
            return None

        # Pattern matching across experiments
        if self._check_consistency(experiment_results):
            principle = Principle(
                rule=self._formulate_rule(experiment_results),
                conditions=self._extract_conditions(experiment_results),
                effectiveness=self._measure_effectiveness(experiment_results),
                domain=self._infer_domain(experiment_results),
                date_extracted=datetime.now()
            )
            self.principles.append(principle)
            self.save()
            return principle
        return None

    def _check_consistency(self, experiment_results: List[Dict[str, Any]]) -> bool:
        """Check if experiments show consistent patterns"""
        if len(experiment_results) < 3:
            return False

        # Check for consistent performance improvements
        performances = [exp.get('fitness_improvement', 0) for exp in experiment_results]
        if not performances:
            return False

        # Require at least 70% of experiments show positive improvement
        positive_ratio = sum(1 for p in performances if p > 0) / len(performances)
        return positive_ratio >= 0.7

    def _formulate_rule(self, experiment_results: List[Dict[str, Any]]) -> str:
        """Formulate a general rule from experiment patterns"""
        # Analyze common parameters that led to success
        successful_experiments = [exp for exp in experiment_results if exp.get('fitness_improvement', 0) > 0]

        if not successful_experiments:
            return "No clear rule identified"

        # Find common parameters
        common_params = self._find_common_parameters(successful_experiments)

        if 'plasticity' in common_params:
            return f"Higher plasticity ({common_params['plasticity']:.2f}) improves learning in {self._infer_domain(experiment_results)} tasks"
        elif 'learning_rate' in common_params:
            return f"Learning rate around {common_params['learning_rate']:.4f} optimizes convergence in {self._infer_domain(experiment_results)} tasks"
        elif 'architecture' in common_params:
            return f"{common_params['architecture']} architecture enhances performance in {self._infer_domain(experiment_results)} tasks"
        else:
            return f"Parameter combination {common_params} leads to improved performance in {self._infer_domain(experiment_results)} tasks"

    def _extract_conditions(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract conditions under which the principle applies"""
        conditions = {}

        # Analyze task characteristics
        task_types = [exp.get('task_type', 'unknown') for exp in experiment_results]
        if len(set(task_types)) == 1:
            conditions['task_type'] = task_types[0]

        # Analyze environment complexity
        complexities = [exp.get('complexity', 0) for exp in experiment_results]
        if complexities:
            conditions['complexity_range'] = {
                'min': min(complexities),
                'max': max(complexities),
                'mean': np.mean(complexities)
            }

        # Analyze agent capabilities
        capabilities = []
        for exp in experiment_results:
            if 'plasticity_enabled' in exp:
                capabilities.append('plasticity' if exp['plasticity_enabled'] else 'fixed')
            if 'meta_learning' in exp:
                capabilities.append('meta_learning' if exp['meta_learning'] else 'standard')

        if capabilities:
            conditions['agent_capabilities'] = list(set(capabilities))

        return conditions

    def _measure_effectiveness(self, experiment_results: List[Dict[str, Any]]) -> float:
        """Measure how effective the principle is"""
        improvements = [exp.get('fitness_improvement', 0) for exp in experiment_results]
        if not improvements:
            return 0.0

        # Effectiveness is the average improvement, normalized
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements) if len(improvements) > 1 else 0

        # Penalize high variance (inconsistent results)
        consistency_penalty = std_improvement / (abs(avg_improvement) + 1e-6)
        effectiveness = max(0, avg_improvement * (1 - min(0.5, consistency_penalty)))

        return float(effectiveness)

    def _infer_domain(self, experiment_results: List[Dict[str, Any]]) -> str:
        """Infer the domain from experiment results"""
        domains = [exp.get('domain', 'general') for exp in experiment_results]
        if domains:
            # Return most common domain
            from collections import Counter
            return Counter(domains).most_common(1)[0][0]
        return 'general'

    def _find_common_parameters(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find parameters that are common across successful experiments"""
        if not experiments:
            return {}

        # Collect all parameters
        all_params = {}
        for exp in experiments:
            for key, value in exp.items():
                if key not in ['fitness', 'fitness_improvement', 'timestamp', 'experiment_id']:
                    if key not in all_params:
                        all_params[key] = []
                    all_params[key].append(value)

        # Find parameters with low variance (consistent values)
        common_params = {}
        for param, values in all_params.items():
            if len(values) > 1:
                try:
                    if isinstance(values[0], (int, float)):
                        # For numeric values, check coefficient of variation
                        mean_val = np.mean(values) if values else 0.0
                        std_val = np.std(values) if values else 0.0
                        if mean_val != 0 and std_val / abs(mean_val) < 0.3:  # CV < 30%
                            common_params[param] = mean_val
                    else:
                        # For categorical values, check if all same
                        if len(set(str(v) for v in values)) == 1:
                            common_params[param] = values[0]
                except:
                    continue

        return common_params

    def query(self, question: str) -> List[Dict[str, Any]]:
        """Query knowledge base"""
        # Simple semantic search through theories/principles
        results = []

        # Search theories
        for theory in self.theories:
            relevance = self._calculate_relevance(question, theory.statement)
            if relevance > 0.3:  # Relevance threshold
                results.append({
                    'type': 'theory',
                    'content': theory,
                    'relevance': relevance,
                    'confidence': theory.confidence
                })

        # Search principles
        for principle in self.principles:
            relevance = self._calculate_relevance(question, principle.rule)
            if relevance > 0.3:
                results.append({
                    'type': 'principle',
                    'content': principle,
                    'relevance': relevance,
                    'confidence': principle.confidence
                })

        # Sort by relevance and confidence
        results.sort(key=lambda x: x['relevance'] * x['confidence'], reverse=True)

        return results[:10]  # Return top 10 results

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate semantic relevance between query and text"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        # Simple Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def add_counter_example(self, theory_statement: str, counter_example: Dict[str, Any]):
        """Add a counter example that challenges a theory"""
        # Find the theory
        theory = None
        for t in self.theories:
            if t.statement == theory_statement:
                theory = t
                break

        if theory is None:
            raise ValueError(f"Theory not found: {theory_statement}")

        # Add counter example
        counter_example['date_found'] = datetime.now()
        if theory.counter_examples is None:
            theory.counter_examples = []
        theory.counter_examples.append(counter_example)

        # Reduce theory confidence
        theory.confidence = max(0.1, theory.confidence * 0.9)  # Reduce by 10%

        self.save()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_theories': len(self.theories),
            'total_principles': len(self.principles),
            'total_strategies': len(self.strategies),
            'total_counter_examples': len(self.counter_examples),
            'domains': list(set([t.domain for t in self.theories] + [p.domain for p in self.principles])),
            'average_theory_confidence': np.mean([t.confidence for t in self.theories]) if self.theories else 0.0,
            'average_principle_effectiveness': np.mean([p.effectiveness for p in self.principles]) if self.principles else 0.0
        }

    def close(self):
        """Close database connections"""
        if self.use_sqlite and hasattr(self, 'conn'):
            self.conn.close()
