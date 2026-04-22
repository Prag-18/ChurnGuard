# ChurnGuard Phase 4: Business Intelligence & Strategy Report

## SECTION A — Executive Summary
- **Revenue at Risk:** We have identified 4,769 customers with a high probability of churn, representing **$5,288,479.34** in projected lifetime value losses.
- **Critical Segment:** The **'New & Unsettled'** segment is currently the most volatile, with a churn rate of **70.4%**.
- **Primary Churn Driver:** Low 'engagement_score' is the #1 predictor across all segments, indicating that declining product usage is the most reliable leading indicator of churn.
- **Cohort Trends:** Churn rates have shown a steady increase in more recent signup cohorts, suggesting a potential decline in customer quality or onboarding effectiveness.
- **Strategic Opportunity:** Successfully retaining just 10% of the currently at-risk population would recover approximately **$528,847.93** in revenue.

## SECTION B — Segment Retention Playbook

### Segment: At-risk Premium
- **Churn rate:** 18.8% | **Avg CLV:** $1,656.84
- **Top drivers:** Engagement Score, Tenure
- **Recommended Action:** Proactive outreach from account managers with exclusive contract upgrade offers (20% loyalty discount).
- **Urgency Level:** **CRITICAL**

### Segment: New & Unsettled
- **Churn rate:** 70.4% | **Avg CLV:** $1,413.32
- **Top drivers:** Engagement Score, Month-to-month Contract
- **Recommended Action:** Immediate onboarding check-in and 'first 90 days' rewards program to encourage long-term commitment.
- **Urgency Level:** **HIGH**

### Segment: Loyal Mid-tier
- **Churn rate:** 15.6% | **Avg CLV:** $2,109.48
- **Top drivers:** Engagement Score, Complaints Count
- **Recommended Action:** Automated usage-based feature recommendations and proactive technical health checks.
- **Urgency Level:** **MEDIUM**

### Segment: Veteran Champion
- **Churn rate:** 1.0% | **Avg CLV:** $1,191.11
- **Top drivers:** Engagement Score, Tenure
- **Recommended Action:** Invitation to VIP feedback panels and early access to new service rollouts (Low touch, high value).
- **Urgency Level:** **LOW**

## SECTION C — Revenue Impact Analysis
- **Total Customers at Risk (>50% Prob):** 4,769
- **Total Value of At-Risk Customers (CLV):** $5,288,479.34
- **Revenue Saved (if 10% improvement):** $528,847.93
- **Top 3 Strategic Levers:**
  1. **Contract Migration:** Converting Month-to-Month users to Annual/Two-year contracts.
  2. **Engagement Boosting:** Incentivizing usage of digital features (Online Backup, Tech Support).
  3. **Service Quality:** Prioritizing ticket resolution for high-CLV customers with multiple complaints.

## SECTION D — Advanced Insight: The 'Tenure Cliff'
- **What the SHAP Model Tells Us:** Our AI analysis reveals a critical inflection point in customer lifecycle. For the first 6 months, 'Tenure' is actually a *risk factor* (positive SHAP values)—meaning new customers are naturally flighty.
- **The Turning Point:** At exactly Month 7, the SHAP values drop below zero, meaning tenure suddenly becomes a *protective factor*. Once they survive the first half-year, their natural probability of staying jumps significantly.
- **Action for the Retention Team TODAY:** There is a 'Danger Zone' in the first 6 months. Do not wait for them to churn. Set up an automated **'6-Month Milestone' reward program** today. Getting a customer over that specific cliff is the highest ROI action possible.

## SECTION E — Model Integrity & Segmentation Logic
- **Data Science Note:** Our segmentation uses 'Power Transformation' to handle skewed financial data, ensuring that segments represent distinct behaviors rather than just outliers.
- **Cluster Distribution:** You may notice unequal segment sizes (e.g., 'Mass Market' vs 'Niche VIPs'). This is a deliberate reflection of our business reality.
- **Interpretability:** Each cluster was vetted for business logic. Small clusters represent high-impact opportunities (e.g., At-risk Premium), while larger clusters represent our stable customer base.

## SECTION F — Model Confidence Assessment
- **High-Confidence Zone:** The model is highly certain (Prob <15% or >85%) for **63.5%** of the customer base.
- **Uncertainty Zone:** There are **1,699** customers in the 'borderline' zone (40-60% churn probability).
- **Human Review Threshold:** We recommend manual intervention and account review for any customer with a churn probability > 0.40 who also belongs to the 'At-risk Premium' segment.
