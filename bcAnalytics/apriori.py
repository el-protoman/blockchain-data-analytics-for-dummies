from efficient_apriori import apriori
transactions=[('eggs', 'bacon', 'bread','milk'),('eggs','bread','milk'),('bacon','bread','milk'),('eggs','bacon','bread'),('eggs','bacon')]
itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=1)

print(itemsets)

rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)
