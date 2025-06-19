#_Types.h

#estas variables son de _rand_MT
import random
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


packet_size = 1518
streams = 32
load_dmb = 0.4
ONU_bit_rate = 1e9 
time_limit = 0.002

num_ont = 16  # Número de ONTs a simular

# Definiciones de tipos en Python
rnd_real_t = float
rnd_int_t = int

SMALL_VAL = 1.0 / 0xFFFFFFFF
# aqui acaban las variables de _rand_MT

#estas variables son de trace
# Definición de tipos
pct_size_t = int
bytestamp_t = float

# Valores por defecto
BYTE_SIZE = 8
PREAMBLE = 8

MIN_PCT = 64
MAX_PCT = 1518

PRIOR_L0 = 0  # lowest priority
PRIOR_L1 = 1  
PRIOR_L2 = 2  
PRIOR_L3 = 3  # highest priority

NULL = None
# aqui acaban las variables de _TRACE

#VARIABLES DE SOURCE
MIN_BURST = 1
MIN_ALPHA = 1.0
MAX_ALPHA = 2.0
#FINAL DE LAS VARIABLES DE SOURCE

#estas variables son de TYPES
# Abstracciones de tipos de datos
int8u = int
int8s = int

int16u = int
int16s = int

int32u = int
int32s = int

int64u = int
int64s = int

BYTE = int
CHAR = str

FLOAT = float
DOUBLE = float

# Funciones utilitarias
def round(val):
    return int(val + 0.5)

def MAX(x, y):
    return x if x > y else y

def MIN(x, y):
    return x if x < y else y

def SWAP(x, y):
    return y, x
    
#***************************
#_rand_MT.h
#***************************

# Sembrar el generador de números aleatorios
def _seed():
    random.seed()

# Generar un número real uniforme en [0, 1]
def _uniform_real_0_1():
    return random.random()

# Generar un número real uniforme en [0, 1)
def _uniform_real_0_X1():
    return random.uniform(0, 1)

# Generar un número real uniforme en (0, 1]
def _uniform_real_X0_1():
    return 1.0 - _uniform_real_0_X1()

# Generar un número real uniforme en un rango dado
def _uniform_real_(low, hi):
    return random.uniform(low, hi)

# Generar un número entero uniforme en un rango dado
def _uniform_int_(low, hi):
    return random.randint(low, hi)

# Generar un número aleatorio con distribución exponencial
def _exponent_():
    return -math.log(_uniform_real_X0_1())

# Generar un número aleatorio con distribución de Pareto
def _pareto_(shape):
    return (1.0 / _uniform_real_X0_1()) ** (1.0 / shape)
#***************************
#***************************


#***************************
#trace.h
#***************************
#***************************

# Clase Trace
class Trace:
    def __init__(self, sid=0, qid=PRIOR_L0, bs=0.0, ps=0):
        self.SourceID = sid
        self.ColaID = qid
        self.ByteStamp = bs
        self.PacketSize = ps
       

    def Append(self, free_stamp):
        self.ByteStamp = max(self.ByteStamp, free_stamp + self.PacketSize)
        return self.ByteStamp

    def __str__(self):
        return f"{self.SourceID}   {self.ColaID}   {self.ByteStamp}   {self.PacketSize}"
#***************************
#***************************

#***************************
#***************************
#link

class Linkable:
    def __init__(self, next=None):
        self.next = next

    def get_next(self):
        return self.next

    def insert(self, prv, nxt):
        if prv:
            prv.next = self
        self.next = nxt

    def remove(self, prv):
        if prv:
            prv.next = self.next


class DLinkable:
    def __init__(self, prev=None, next=None):
        self.prev = prev
        self.next = next

    def GetNext(self):
        return self.next

    def GetPrev(self):
        return self.prev

    def InsertAfter(self, ptr):
        self.insert(ptr, ptr.next if ptr else None)

    def InsertBefore(self, ptr):
        self.insert(ptr.next if ptr else None, ptr)

    def Insert(self, prv, nxt):
        self.prev = prv
        if prv:
            prv.next = self
        self.next = nxt
        if nxt:
            nxt.prev = self

    def Remove(self):
        if self.prev:
            self.prev.next = self.next
        if self.next:
            self.next.prev = self.prev


class DList:
    def __init__(self):
        self.Clear()

    def Clear(self):
        self.head = self.tail = None
        self.count = 0

    def GetCount(self):
        return self.count

    def GetHead(self):
        return self.head

    def GetTail(self):
        return self.tail

    def Append(self, ptr):
        self.insert(self.tail, ptr, None)

    def InsertHead(self, ptr):
        self.insert(None, ptr, self.head)

    def Insert(self, prv, ptr, nxt):
        if ptr:
            if prv is None or nxt == self.head:
                self.head = ptr
            if nxt is None or prv == self.tail:
                self.tail = ptr
            ptr.Insert(prv, nxt)
            self.count += 1

    def RemoveHead(self):
        return self.Remove(self.head)

    def RemoveTail(self):
        return self.Remove(self.tail)

    def Remove(self, ptr):
        if ptr:
            if ptr == self.head:
                self.head = self.head.GetNext()
            if ptr == self.tail:
                self.tail = self.tail.GetPrev()
            ptr.Remove()
            self.count -= 1
        return ptr

    def Iterator(self, func):
        ptr = self.head
        while ptr:
            func(ptr)
            ptr = ptr.get_next()


#***************************
#***************************

#***************************
#***************************
#source
class Source(DLinkable):
    MIN_BURST = 1
    MIN_ALPHA = 1.0
    MAX_ALPHA = 2.0

    def __init__(self, id, prior, pct_sz, preamble):
        super().__init__()
        self.ID = id
        self.Priority = prior
        self.PctSize = pct_sz
        self.Preamble = preamble
        self.PctSpace = self.PctSize + self.Preamble
        self.Elapsed = 0.0
        self.BurstSize = 0

    def Reset(self):
        burst_size = self.GetBurstSize() * self.PctSpace
        period = burst_size + self.GetGapSize()
        start_time = _uniform_real_(0, period)
        if start_time > burst_size:
            self.BurstSize = self.GetBurstSize()
            self.Elapsed = period - start_time
        else:
            self.BurstSize = int((burst_size - start_time) / self.PctSpace + 1)
            self.Elapsed = 0.0

    def GetID(self):
        return self.ID

    def GetPriority(self):
        return self.Priority

    def GetArrival(self):
        return self.Elapsed

    def GetPctSize(self):
        return self.PctSize

    def GetTrace(self):
        return Trace(self.ID, self.Priority, self.Elapsed, self.PctSize)

    def ExtractPacket(self, trc=None):
        if trc is not None:
            trc.SourceID = self.ID
            trc.ColaID = self.Priority
            trc.ByteStamp = self.Elapsed
            trc.PacketSize = self.PctSize
        if self.BurstSize == 0:
            self.BurstSize = self.GetBurstSize()
            self.Elapsed += self.GetGapSize()
        self.BurstSize -= 1
        self.Elapsed += self.PctSpace


class SourcePareto(Source):
    MIN_BURST = 1
    MIN_ALPHA = 1.0
    MAX_ALPHA = 2.0

        
    def __init__(self, id, prior, pct_sz, preamble, load, on_shape, off_shape):
        super().__init__(id, prior, pct_sz, preamble)
        self.ONShape = on_shape
        self.OFFShape = off_shape
        self.SetInRange(self.ONShape, self.MIN_ALPHA, self.MAX_ALPHA)
        self.SetInRange(self.OFFShape, self.MIN_ALPHA, self.MAX_ALPHA)
        self.SetLoad(load)
        self.Reset()
        
    def SetGap(self, gap):
        return PREAMBLE if gap < PREAMBLE else gap
        
    def SetLoad(self, load):
        self.SetInRange(load, 0.0, 1.0)
        on_coef = (1.0 - pow(SMALL_VAL, 1.0 - 1.0/self.ONShape)) / (1.0 - 1.0/self.ONShape)
        off_coef = (1.0 - pow(SMALL_VAL, 1.0 - 1.0/self.OFFShape)) / (1.0 - 1.0/self.OFFShape)
        self.MinGap = self.SetGap((on_coef/off_coef) * self.MIN_BURST * (self.PctSize/load - self.PctSpace))
    
    def GetBurstSize(self):
        return round(_pareto_(self.ONShape) * self.MIN_BURST)
        

    def GetGapSize(self):
        return _pareto_(self.OFFShape) * self.MinGap
       
       

    def SetInRange(self, x, y, z):
        if x <= y or x >= z:
            x = (y + z) / 2
#***************************
#***************************


#***************************
#***************************
#aggreg
class Generator:
    def __init__(self):
        self.SRC = DList()  # Lista doblemente enlazada de fuentes
        self.NextPacket = Trace()  # Paquete futuro
        self.TotalBytes = 0.0  # Total de bytes generados
        self.TotalPackets = 0  # Total de paquetes generados
        self.Elapsed = 0.0  # Tiempo transcurrido en unidades de ByteTime desde el inicio del rastreo
        self.ByteStamp = 0
        self.Preamble = PREAMBLE
        self.Reset()

    def InsertInOrder(self, pSrc):
        arrival = pSrc.GetArrival()
        pPrv = None
        pNxt = self.SRC.GetHead()

        while pNxt and arrival > pNxt.GetArrival():
            pPrv = pNxt
            pNxt = pNxt.GetNext()

        self.SRC.Insert(pPrv, pSrc, pNxt)
        self.NextPacket = self.SRC.GetHead().GetTrace()
        self.ByteStamp = self.NextPacket.Append(self.Elapsed + self.Preamble)
        return pSrc

    def Reset(self):
        self.TotalBytes = 0.0
        self.Elapsed = 0.0
        self.TotalPackets = 0

        lst = self.SRC
        self.SRC.Clear()

        ptr = lst.RemoveHead()
        while ptr is not None:
            ptr.Reset()
            self.InsertInOrder(ptr)
            ptr = lst.RemoveHead()

    def GetPackets(self):
        return self.TotalPackets

    def GetBytes(self):
        return self.TotalBytes

    def GetTime(self):
        return self.Elapsed

    def GetByteStamp(self):
        return self.ByteStamp

    def GetLoad(self):
        return self.TotalBytes / self.Elapsed

    def GetSources(self):
        return self.SRC.GetCount()

    def AddSource(self, pSrc):
        if pSrc:
            self.InsertInOrder(pSrc)

    def RemoveSource(self, pSrc=None):
        if pSrc is None:
            return self.SRC.RemoveHead()
        else:
            self.SRC.Remove(pSrc)

    def SetLoad(self, load):
        count = self.SRC.GetCount()
        if count > 0:
            load /= count
            ptr = self.SRC.GetHead()
            while ptr:
                ptr.SetLoad(load)
                ptr = ptr.GetNext()

    def PeekNextPacket(self):
        return self.NextPacket

    def GetNextPacket(self):
        trc = self.NextPacket
        if self.SRC.GetHead():
        
            self.Elapsed = self.NextPacket.ByteStamp
            self.TotalBytes += self.NextPacket.PacketSize
            self.TotalPackets += 1

        
            pSrc = self.SRC.RemoveHead()
            pSrc.ExtractPacket()
            self.InsertInOrder(pSrc)

        return trc
#***************************
#***************************
class ONT:
    def __init__(self, id, bit_rate, packet_size, load, streams):
        self.ID = id
        self.BitRate = bit_rate
        self.PacketSize = packet_size
        self.Load = load
        self.Streams = streams
        self.Generator = Generator()  
        self.Cola = []  
        self.MaxColaSize = 10e8
        self.CurrentColaSize = 0  
        self.DiscardedPackets = 0
        self.Timestamps = []
        self.LastProcessedTime = 0.0

        for src in range(self.Streams):
            self.Generator.AddSource(SourcePareto(src, 0, self.PacketSize, 0, self.Load / self.Streams, 1.4, 1.2))

    def GetNextPacket(self):
        packet = self.Generator.GetNextPacket()
        self.AddToCola(packet)
        self.Timestamps.append(self.GetCurrentTime())  
        return packet

    def AddToCola(self, packet):
        if self.CurrentColaSize + packet.PacketSize <= self.MaxColaSize:
            self.Cola.append(packet)
            self.CurrentColaSize += packet.PacketSize
        else:
            self.DiscardedPackets += 1
    
    def GetColaSize(self):
        return len(self.Cola)
    
    def GetDiscardedPackets(self):
        return self.DiscardedPackets
    
    def GetCurrentTime(self):
        return self.Generator.GetTime()
    
    def GetByteStamp(self):
        return self.Generator.GetByteStamp()

    def GetTotalLoad(self):
        elapsed_time = self.GetCurrentTime()
        if elapsed_time > 0:
            return (self.Generator.GetBytes() * 8) / (elapsed_time * self.BitRate)
        return 0.0

    def Reset(self):
        self.Generator.Reset()
        self.Cola.clear() 
        self.Timestamps.clear()

    def __str__(self):
        return f"ONT {self.ID}, Paquetes totales={self.Generator.GetPackets()},Paquetes descartados={self.GetDiscardedPackets()}, Bytes totales={self.Generator.GetBytes()}, Cola={len(self.Cola)} paquetes"

# Ejemplo de uso MAIN
#ESTO NO SE DEBERÍA PONER EN EL MAIN, ES UN EJEMPLO DE USO DEL TRACE

def simular_trafico(onts=None):

    if onts is None:
        onts = [ONT(id=i, bit_rate=ONU_bit_rate, packet_size=packet_size, load=load_dmb, streams=streams) for i in range(num_ont)] 

    suma_tam_packets = [0] * 16
    total_loads = [0.0] * 16
    trafico_entrada_por_ciclo = [0]*16
   
    
    for ont in onts: 
        current_time = 0.0

        while current_time <= time_limit:  
            trc = ont.GetNextPacket()
            tiempo = ont.GetCurrentTime()
            byte_stamp = ont.GetByteStamp()

            intervalo_actual = byte_stamp - tiempo
            interarrivaltime = intervalo_actual * 8 / ONU_bit_rate
            suma_tam_packets[ont.ID] += trc.PacketSize
            current_time += interarrivaltime
    
            total_loads[ont.ID] = (suma_tam_packets[ont.ID] * 8) / (current_time * ONU_bit_rate)  
            trafico_entrada_por_ciclo[ont.ID] = suma_tam_packets[ont.ID]*8
            
    return  trafico_entrada_por_ciclo, onts


def main():
    simular_trafico()

if __name__ == "__main__":
    main()