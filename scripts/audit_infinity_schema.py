"""Offline Parquet inspection only; not training/tokenization glue."""
import json, pathlib, collections
import pyarrow.parquet as pq
root=pathlib.Path("D:/Datasets/Infinity-Instruct")
manifest=json.loads((root/"download_manifest.json").read_text())
report={"revision":manifest["revision"],"sampling":"first 32 rows of first batch of each shard; biased diagnostic, not random/full scan","shards":[],"schemas":{},"sample_rows":0,"roles":{},"languages":{},"source_counts":{},"conversation_lengths":[],"empty_text":0,"malformed_messages":0,"role_sequences":{}}
roles=collections.Counter();langs=collections.Counter();sources=collections.Counter();seq=collections.Counter()
for entry in manifest["files"]:
 if not entry["path"].endswith(".parquet"):continue
 p=root/entry["path"];f=pq.ParquetFile(p);schema=str(f.schema_arrow.remove_metadata());report["schemas"][schema]=report["schemas"].get(schema,0)+1
 report["shards"].append({"path":entry["path"],"rows":f.metadata.num_rows,"row_groups":f.metadata.num_row_groups,"size_matches":p.stat().st_size==entry["bytes"]})
 rows=next(f.iter_batches(batch_size=32)).to_pylist()
 for row in rows:
  report["sample_rows"]+=1;langs[str(row.get("langdetect","MISSING"))]+=1;sources[str(row.get("source","MISSING"))]+=1
  messages=row.get("conversations",row.get("messages",[]));order=[];length=0
  if not isinstance(messages,list):report["malformed_messages"]+=1;continue
  for message in messages:
   if not isinstance(message,dict):report["malformed_messages"]+=1;continue
   role=message.get("from",message.get("role","MISSING"));text=message.get("value",message.get("content"));roles[str(role)]+=1;order.append(str(role))
   if not isinstance(text,str):report["malformed_messages"]+=1
   else:length+=len(text);report["empty_text"]+=int(not text.strip())
  seq[" -> ".join(order)]+=1;report["conversation_lengths"].append(length)
report["roles"]=dict(roles);report["languages"]=dict(langs);report["source_counts"]=dict(sources);report["role_sequences"]=dict(seq)
lengths=sorted(report.pop("conversation_lengths"));report["sample_character_lengths"]={"min":lengths[0],"median":lengths[len(lengths)//2],"p95":lengths[int(len(lengths)*.95)],"max":lengths[-1]}
report["total_rows_metadata"]=sum(s["rows"] for s in report["shards"])
out=pathlib.Path("INFINITY_SCHEMA_AUDIT.json");out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ["shards","source_counts","role_sequences"]},ensure_ascii=True,indent=2))
print("role_sequences",seq.most_common(8))
