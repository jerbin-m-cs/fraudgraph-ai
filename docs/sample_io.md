txn_id:             RING_00_00
account:            CUST103041
device:             CARD_INJECTED_01
merchant:           MER100466
amount:             ₹56,118
timestamp:          2024-06-15 02:00:00
location:           Lakshadweep
Is_International:   0
Transaction_Status: Successful
Card_Status:        Active

## Output — After the engine scores it

risk_score:    100 / 100
decision:      BLOCK
raw_score:     110

reasons:
  +50  Ring signature: card shared by 5 accounts
  +25  Card CARD_INJECTED_01 shared by 5 accounts
  +20  New card (CARD_INJECTED_01)
  +15  New merchant (MER100466)
  +10  Off-hours transaction (02:xx)

graph_result:
  ring_id:       R1
  size:          5 accounts
  shared_card:   CARD_INJECTED_01
  accounts:      [CUST103041, CUST103073, CUST108403, CUST109187, CUST114608]

recommended_action:
  BLOCK — freeze account, flag card for replacement, notify customer

## Interpretation

The engine detects three layers of evidence:

1. Individual anomalies — new card, new merchant, off-hours transaction
2. Coordination signal — the same card is used by 5 different accounts
3. Ring confirmation — graph analysis groups the 5 accounts into a single ring R1

The final score of 100/100 reflects the combined evidence. The BLOCK decision
is deterministic and fully explained by the reason strings — no black-box
model is used.
