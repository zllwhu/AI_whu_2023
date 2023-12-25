import math
import random


class Life:
    """ 个体类 """

    def __init__(self, gene):
        self.gene = gene  # 个体的基因
        self.score = -1  # 个体的适应度得分


class GeneticAlgorithm:
    """ 遗传算法类 """

    def __init__(self, across_rate, mutation_rate, life_count, gene_length, match_fun):
        self.across_rate = across_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.life_count = life_count  # 种群中个体数量
        self.gene_length = gene_length  # 基因长度
        self.match_fun = match_fun  # 适配函数
        self.lives = []  # 种群
        self.best = None  # 本代目最佳个体
        self.generation = 1  # 代目
        self.cross_count = 0  # 交叉计数
        self.mutation_count = 0  # 变异计数
        self.bounds = 0.0  # 种群适配函数值之和
        self.mean = 1.0  # 种群平均适配函数值
        self.init_population()

    def init_population(self):
        for i in range(self.life_count):
            gene = [x for x in range(self.gene_length)]
            random.shuffle(gene)
            self.lives.append(Life(gene))

    def judge(self):
        self.bounds = 0.0
        self.best = self.lives[0]
        for life in self.lives:
            life.score = self.match_fun(life)
            self.bounds += life.score
            if self.best.score < life.score:
                self.best = life
        self.mean = self.bounds / self.life_count

    def cross(self, parent1, parent2, max_test=120):
        test = 0
        while True:
            new_gene = []
            index1 = random.randint(0, self.gene_length - 1)
            index2 = random.randint(index1, self.gene_length - 1)
            cross_gene = parent2.gene[index1:index2]
            i_flag = 0
            for g in parent1.gene:
                if i_flag == index1:
                    new_gene.extend(cross_gene)
                    i_flag += 1
                if g not in cross_gene:
                    new_gene.append(g)
                    i_flag += 1
            if self.match_fun(Life(new_gene)) > max(self.match_fun(parent1), self.match_fun(parent2)):
                self.cross_count += 1
                return new_gene
            if test > max_test:
                self.cross_count += 1
                return new_gene
            test += 1

    def mutation(self, child):
        new_gene = child.gene[:]
        index1 = random.randint(0, self.gene_length - 1)
        index2 = random.randint(0, self.gene_length - 1)
        new_gene[index1], new_gene[index2] = new_gene[index2], new_gene[index1]
        if self.match_fun(Life(new_gene)) > self.match_fun(child):
            self.mutation_count += 1
            return new_gene
        else:
            rate = random.random()
            if rate < math.exp(-10 / math.sqrt(self.generation)):
                self.mutation_count += 1
                return new_gene
            return child.gene

    def get_one(self):
        r = random.uniform(0, self.bounds)
        for life in self.lives:
            r -= life.score
            if r <= 0:
                return life

    def new_child(self):
        parent1 = self.get_one()
        rate = random.random()
        if rate < self.across_rate:
            parent2 = self.get_one()
            gene = self.cross(parent1, parent2)
        else:
            gene = parent1.gene
        rate = random.random()
        if rate < self.mutation_rate:
            gene = self.mutation(Life(gene))
        return Life(gene)

    def next_iter(self):
        self.judge()
        new_lives = [self.best]
        while len(new_lives) < self.life_count:
            new_lives.append(self.new_child())
        self.lives = new_lives
        self.generation += 1
