# %%
import torch
import json
import math
import os
from collections import defaultdict
# %%
with open("llama_outputs_zeroshot_greedy.txt", "r") as f:
    lines = f.readlines()
with open("llama_truths.txt", "r") as f:
    truths = [json.loads(x) for x in f.readlines()]

categorical_fields = [
  "phone.oem",
  "phone.network_edge"
]

text_fields = [
  "phone.model",
]
# %%
n_errors = 0
samples = []
valid_inputs = 0
for idx, line in enumerate(lines):
  # truth = truths[idx]
  print(idx, line)
  truth, output = line.split(" <|> ")
  truth = json.loads(truth)
  assert truth == truths[idx]
  field_of_interest = list(truth.keys())[-1]
  if isinstance(truth[field_of_interest], float) and math.isnan(truth[field_of_interest]):
    continue

  # cut outputs after the last field is predicted and remove the last comma
  # last key of truth:
  last_key = list(truth.keys())[-1]
  # find that key in outputs
  last_key_idx = output.find(last_key)
  # from the on, go until the next comma
  last_comma_idx = output.find(",", last_key_idx)
  output = output[:last_comma_idx] + "}"

  valid_inputs += 1
  try:
    parsed_output = json.loads(output)
    assert list(truth.keys()) == list(parsed_output.keys())
    if field_of_interest in categorical_fields + text_fields:
      assert isinstance(parsed_output[field_of_interest], str)
    else:
      assert isinstance(parsed_output[field_of_interest], float)
      assert not math.isnan(parsed_output[field_of_interest])
    samples += [(truth, parsed_output)]
  except json.JSONDecodeError as e:
    n_errors += 1
  except AssertionError:
    n_errors += 1

print(f"error rate {n_errors / valid_inputs:.1e}")
# %%
def get_metrics(data):
  # others are numerical
  metrics = defaultdict(list)
  for truth, output in data:
    field = list(truth.keys())[-1]
    if field in categorical_fields:
      assert not isinstance(truth[field], float) and not isinstance(output[field], float)
      metrics[field].append(float(truth[field].strip() == output[field].strip()))
    elif field in text_fields:
      assert not isinstance(truth[field], float) and not isinstance(output[field], float)
      truth_field = set(truth[field].strip().split())
      output_field = set(output[field].strip().split())
      metrics[field].append(len(truth_field & output_field) / len(truth_field | output_field))
    else:
      metrics[field].append((output[field] - truth[field])**2)
  return metrics

metrics = get_metrics(samples)
# %%
for key in metrics:
  metric = [x for x in metrics[key] if not math.isnan(x)]
  if key in categorical_fields + text_fields:
    print(f"{key}: {1 - sum(metric) / len(metric):.3f}")
  else:
    print(f"{key}: {(sum(metric) / len(metric))**.5:.3f}")
# %%