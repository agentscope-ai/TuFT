# Sample outputs — before vs. after OPD

Real, greedy-decoded generations from this example: a **Qwen3-1.7B** student, distilled on-policy from a
**few-shot Qwen3-1.7B teacher** (`train.py`, 16 steps, `lr=1e-4`, rank 16, single A100-40GB). Both columns
use the **same bare prompt** (just the question) — the few-shot examples are baked into the student's LoRA,
not shown at inference.

On the held-out GSM8K slice (seed 0, 80 problems): **bare student 62.5% → after OPD 71.2%**, with
**13 problems flipping wrong → right** and 6 going the other way (net +7). A representative selection is below.

> These are captured from one real run. Sampling and training have run-to-run variance, so the exact
> problems, wording, and numbers will differ if you re-run — the *pattern* (ornate-and-wrong → short-and-right)
> is what reproduces.

The bare model fails in two recurring ways. Both are fixed the same way: OPD makes the student commit to the
teacher's short, complete arithmetic instead of an open-ended essay.

---

## A. Dropped or mishandled a constraint

### Susan's earnings — ignores the half-hour pause  (answer: 75)

> Susan earns $5 every 10 minutes for an online task she does. If she works between 8 a.m. and 11 a.m. and
> pauses in between for half an hour, how much money does she earn for the online task?

**Before (wrong → $90):** lists the pause in "Given", then never subtracts it.
```text
Let's break down the problem step by step.

### Given:
- Susan earns **$5 every 10 minutes**.
- She works **between 8 a.m. and 11 a.m.**.
- She **pauses for half an hour** in between.

### Step 1: Determine the total time she works
From 8 a.m. to 11 a.m. is:
$$
11 - 8 = 3 \text{ hours} = 180 \text{ minutes}
$$

### Step 2: Determine how many 10-minute intervals fit into 180 minutes
$$
\frac{180}{10} = 18 \text{ intervals}
$$

### Step 3: Calculate how much she earns
$$
18 \text{ intervals} \times \$5 = \$90
$$

### Final Answer:
$$
\boxed{90}
$$

Susan earns **$90** for the online task.
```

**After (correct → $75):**
```text
Susan works between 8 a.m. and 11 a.m., which is 3 hours. She pauses for half an hour in between, so she works for 3 - 0.5 = 2.5 hours.

In 10 minutes, she earns $5. There are 10 minutes in 1 hour, so in 2.5 hours, she works 2.5 * 60 = 150 minutes.

150 minutes / 10 minutes = 15 intervals.

15 intervals * $5 = $75.

Susan earns $75 for the online task.

$$
\boxed{75}
$$
```

### Roger's coffee — buys only the expensive brand, but the bare model counts both  (answer: 44)

> Roger goes to the store to buy some coffee. The normal brand of coffee he buys cost $5 per pound. He had
> to buy a more expensive brand that cost 20% more since his favorite brand was sold out. He decides to buy
> a week's worth of coffee and he uses 1 pound of coffee per day. He also decided to buy himself a donut for
> $2. How much did everything cost?

**Before (wrong → $79):** adds the normal-brand cost *and* the expensive-brand cost (he only buys the expensive one).
```text
Let's break this down step by step:

### 1. **Cost of the normal brand of coffee:**
- Price per pound = $5
- He buys **1 pound per day** for **1 week** (7 days)
- Total coffee cost = $5 × 7 = **$35**

### 2. **Cost of the more expensive brand:**
- It's **20% more expensive** than the normal brand
- 20% of $5 = $1
- So, the more expensive brand costs $5 + $1 = **$6 per pound**
- He still buys **1 pound per day** for 7 days
- Total cost for the more expensive brand = $6 × 7 = **$42**

### 3. **Cost of the donut:**
- He buys a donut for **$2**

### 4. **Total cost:**
- Coffee (normal + more expensive) = $35 + $42 = **$77**
- Donut = **$2**
- **Total cost = $77 + $2 = $79**

### ✅ Final Answer: **$79
```

**After (correct → $44):**
```text
The normal coffee costs $5 per pound. The more expensive brand is 20% more expensive, so it costs $5 * 1.20 = $6 per pound.

He uses 1 pound per day for a week, so he needs 7 pounds.

The cost of the coffee is 7 pounds * $6 = $42.

He also buys a donut for $2.

Total cost = $42 + $2 = $44.

The answer is $44.
```

### Tom's trip — counts only 3 days of the second week instead of 7  (answer: 3140)

> Tom went on a two-week-long trip through Europe. In the first 4 days, he traveled 200 kilometers every day,
> and over the next two days, he totaled only 30% of the distance traveled over the first four days. On the
> next day, he wasn't traveling at all. During the second week, he made 300 kilometers every day. How many
> kilometers in total did Tom make during his two-week-long trip?

**Before (wrong):** uses "Days 8 to 10" (3 days) for the second week, then runs out of tokens.
```text
Let's break down Tom's trip into parts and calculate the total distance he traveled.

### **First Week:**

- **Days 1 to 4:**
  He traveled **200 km/day** for 4 days.
  Total = $ 4 \times 200 = 800 $ km

- **Days 5 and 6:**
  He traveled **30% of the distance** from the first 4 days.
  30% of 800 km = $ 0.3 \times 800 = 240 $ km

- **Day 7:**
  He **was not traveling at all**, so distance = 0 km

### **Second Week:**

- **Days 8 to 10:**
  He traveled **300 km/day** for 3 days.
  Total = $ 3 \times 300 = 900 $ km

### **Total Distance:**

- First week: $ 800 + 240 + 0 = 1040 $ km
- Second week: $ 900 $ km
- **Total =
```

**After (correct → 3140):**
```text
First 4 days: 4 days * 200 km/day = 800 km
Next 2 days: 30% of 800 km = 0.3 * 800 = 240 km
Next day: 0 km
Second week: 7 days * 300 km/day = 2100 km
Total: 800 + 240 + 0 + 2100 = 3140 km

Answer: 3140 kilometers.
```

---

## B. Over-elaborated and ran out of the 256-token budget before answering

### Adam's rock stand — goes symbolic with `x` and never plugs in $100  (answer: 16)

> Adam has $100 and wants to spend it to open a rock stand. He can buy rocks for $5 each and sell them for
> $7 each. If he invests all his money in the rock stand but only sells 60% of his inventory, how much money
> does he lose?

**Before (wrong — no final number):**
```text
Let's break down the problem step by step.

### **Step 1: Total money Adam has**
Adam has **$100**.

### **Step 2: Cost of rocks**
- Each rock costs **$5**.
- He wants to buy rocks to open a rock stand.
- He will sell **60%** of his inventory.

So, he needs to buy enough rocks to sell 60% of his inventory.

Let’s assume he buys **$x$** rocks.

- Cost of rocks = $5x$
- He sells 60% of the rocks = $0.6x$
- Selling price per rock = $7
- Total revenue from selling 60% = $7 \times 0.6x = 4.2x$

### **Step 3: Set up the equation**

He spends $5x$ to buy the rocks, and earns $4.2x$ by selling them.

So, the net profit is:

$$
\text{Net profit} = 4.2x - 5x = -0.8x
$$

This means he **loses** $0.8x$.

### **Step 4:
```

**After (correct → 16):**
```text
Adam buys 100 / 5 = 20 rocks. He sells 60% of them: 20 * 0.6 = 12 rocks. He earns 12 * 7 = 84 dollars. He loses 100 - 84 = 16 dollars.

The answer is 16.
```

### Jamal's phone — correct setup, runs out one division short  (answer: 6)

> Jamal's phone can hold 6 times more photographs than can Brittany's phone. The maximum number of
> photographs that Brittany's phone can hold is 50 times more than the number of birds in Jamal's photograph
> of the ducks at the zoo. If Jamal's phone can hold 1800 photographs, how many ducks can be seen in Jamal's
> photograph of ducks at the zoo?

**Before (wrong — cut off at the last step):**
```text
We are given the following information:

1. **Jamal's phone can hold 6 times more photographs than Brittany's phone.**
   Let $ J $ = number of photographs Jamal's phone can hold
   Let $ B $ = number of photographs Brittany's phone can hold
   So, $ J = 6B $

2. **The maximum number of photographs that Brittany's phone can hold is 50 times more than the number of birds in Jamal's photograph of the ducks at the zoo.**
   Let $ D $ = number of birds in Jamal's photograph of the ducks
   So, $ B = 50D $

3. **Jamal's phone can hold 1800 photographs.**
   So, $ J = 1800 $

### Step 1: Use $ J = 6B $ to find $ B $

$$
1800 = 6B \Rightarrow B = \frac{1800}{6} = 300
$$

### Step 2: Use $ B = 50D $ to find $ D $

$$
300 = 50D \Rightarrow D = \frac{300}{
```

**After (correct → 6):**
```text
Let's break it down step by step:

1. Jamal's phone can hold 1800 photographs.
2. Jamal's phone can hold 6 times more photographs than Brittany's phone.
   So, Brittany's phone can hold $ \frac{1800}{6} = 300 $ photographs.
3. Brittany's phone can hold 50 times more than the number of birds in Jamal's photograph of the ducks at the zoo.
   So, the number of birds is $ \frac{300}{50} = 6 $.

The number of ducks in Jamal's photograph is 6.

$$
\boxed{6}
$$
```

### Freda's tomato sauce — runs out before counting the cans  (answer: 12)

> When Freda cooks canned tomatoes into sauce, they lose half their volume. Each 16 ounce can of tomatoes
> that she uses contains three tomatoes. Freda's last batch of tomato sauce made 32 ounces of sauce. How many
> tomatoes did Freda use?

**Before (wrong — cut off mid-derivation):**
```text
Let's break down the problem step by step.

### **Given:**

- **Each 16-ounce can of tomatoes contains 3 tomatoes.**
- **When cooked, the volume of tomatoes loses half.**
- **Freda's last batch of tomato sauce made 32 ounces of sauce.**

### **Step 1: Understand the volume loss**

When Freda cooks the tomatoes into sauce, **half their volume is lost**. So, the **final volume of sauce is half the volume of the tomatoes used**.

Let’s denote the **original volume of tomatoes used** as $ V $.

Then, the **final volume of sauce** is:
$$
\text{Final volume} = \frac{V}{2}
$$

We are told that the final volume of sauce is **32 ounces**. So:
$$
\frac{V}{2} = 32
$$

Multiply both sides by 2:
$$
V = 64 \text{ ounces}
$$

### **Step 2: Find how many cans of tomatoes were used**

Each can contains **16 ounces of tomatoes**, and each can contains **3 tomatoes**.

So, the number of cans used is:
```

**After (correct → 12):**
```text
Each can of tomatoes contains 3 tomatoes.
Each can loses half its volume, so 16 ounces of tomatoes become 8 ounces of sauce.
Freda made 32 ounces of sauce, so she used 32 / 8 = 4 cans.
Each can has 3 tomatoes, so 4 cans * 3 tomatoes = 12 tomatoes.

Answer: 12 tomatoes.
```
