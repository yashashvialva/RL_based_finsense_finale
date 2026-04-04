import random
from typing import Tuple, List, Dict
from finsense.models import Expense

CATEGORIES = {
    "food": {
        "essential": [("Groceries", 200, 1500), ("Rice/Wheat", 400, 1000)],
        "semi-essential": [("Swiggy/Zomato", 150, 800), ("Local Restaurant", 200, 600)],
        "discretionary": [("Fine Dining", 1500, 5000), ("Bar/Pub", 1000, 4000)]
    },
    "transport": {
        "essential": [("Public Transport", 20, 100), ("Metro Pass", 1000, 2000)],
        "semi-essential": [("Uber/Ola Range", 100, 600), ("Auto Rickshaw", 50, 250)],
        "discretionary": [("Flight Booking", 3000, 12000), ("Premium Train", 1000, 3000)]
    },
    "rent": {
        "essential": [("House Rent", 8000, 25000), ("Maintenance Fee", 1000, 3000)]
    },
    "utility": {
        "essential": [("Electricity Bill", 800, 3000), ("Water Bill", 200, 600), ("Internet/Broadband", 600, 1500)]
    },
    "medical": {
        "essential": [("Pharmacy", 200, 2000), ("Doctor Visit", 500, 1500)],
        "semi-essential": [("Health Checkup", 1000, 3000)],
        "discretionary": [("Cosmetic Product", 500, 2000)]
    },
    "entertainment": {
        "discretionary": [("Movie Tickets", 300, 1000), ("Concert", 1500, 5000), ("Streaming Sub", 200, 800)]
    }
}

class ExpenseGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        
    def generate_daily_expenses(self, num_expenses: int) -> List[Expense]:
        expenses = []
        for _ in range(num_expenses):
            # Weighted random choice of category
            category = self.rng.choices(
                list(CATEGORIES.keys()), 
                weights=[40, 20, 5, 10, 5, 20] 
            )[0]
            
            necessity = self.rng.choice(list(CATEGORIES[category].keys()))
            item = self.rng.choice(CATEGORIES[category][necessity])
            
            name = item[0]
            min_amt, max_amt = item[1], item[2]
            amount = round(self.rng.uniform(min_amt, max_amt), 2)
            
            expenses.append(Expense(
                name=name,
                category=category,
                amount=amount,
                necessity_tag=necessity
            ))
        return expenses
        
    def check_shock(self, allow_shocks: bool) -> str:
        if not allow_shocks:
            return "none"
        roll = self.rng.random()
        if roll < 0.05:
            return "emergency_expense"
        elif roll < 0.10:
            return "salary_delay"
        elif roll < 0.15:
            return "discount"
        return "none"
