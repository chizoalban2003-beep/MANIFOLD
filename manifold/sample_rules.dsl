# Example rulebook for non-technical policy editing.
if late_delivery then -£8.20 @target=0.18 @alpha=1.25 @min=0.8 @max=35
if skip_verification then -£5.40 @target=0.22 @alpha=1.05 @min=0.5 @max=32
if low_inventory then -£6.80 @target=0.20 @alpha=1.10 @min=0.7 @max=38
