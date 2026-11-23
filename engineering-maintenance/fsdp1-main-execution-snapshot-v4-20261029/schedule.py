import math
def resolve_schedule(train_records,world_size=8,batch_size=1,grad_accum=4,epochs=2,warmup_ratio=0.03):
 steps_per_epoch=math.ceil(train_records/(world_size*batch_size*grad_accum)); total=steps_per_epoch*epochs
 warmup=max(1,math.ceil(total*warmup_ratio))
 if warmup>=total: raise ValueError('warmup must be below total steps')
 return {'train_records':train_records,'steps_per_epoch':steps_per_epoch,'total_steps':total,'warmup_ratio':warmup_ratio,'warmup_steps':warmup,'effective_global_batch':world_size*batch_size*grad_accum}
