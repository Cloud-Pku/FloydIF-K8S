
from deepfloyd_if.model.respace import space_timesteps

if __name__ == '__main__':
    a =  '10,10,10,10,10,10,10,10,0,0'
    section_counts = [int(x) for x in a.split(',')]
    res = space_timesteps(1000, a)
    res = list(res)
    res.sort()
    print(res)
    print(sum(section_counts), len(section_counts))