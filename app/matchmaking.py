"""
Módulo de simulación de Matchmaking inspirado en CS:GO.
Modela el fenómeno estocástico de solicitud de partida, encolamiento,
generación de perfiles estadísticos basados en las distribuciones ajustadas,
cálculo de compatibilidad por distancias normalizadas y ensamblaje de partidas.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from simulation import generate_discrete, generate_continuous


@dataclass
class Player:
    """Representa a un jugador en la cola de matchmaking."""
    id: int
    skill_rank: float          # Característica derivada de la variable discreta (ej. Rango de habilidad 1-18)
    combat_stat: float         # Característica derivada de la variable continua (ej. Rendimiento/Daño/Tiempo)
    norm_features: np.ndarray  # Vector de características normalizadas en [0, 1]
    arrival_time: float = 0.0
    wait_time: float = 0.0
    match_id: Optional[int] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': f"Jugador #{self.id:03d}",
            'skill_rank': round(float(self.skill_rank), 2),
            'combat_stat': round(float(self.combat_stat), 2),
            'arrival_time': round(float(self.arrival_time), 2),
            'wait_time': round(float(self.wait_time), 2),
            'match_id': self.match_id
        }


@dataclass
class Match:
    """Representa una partida competitiva creada por el sistema."""
    id: int
    players: List[Player]
    creation_time: float
    avg_compatibility: float
    avg_rank: float
    avg_combat: float
    
    def to_dict(self):
        return {
            'id': self.id,
            'match_code': f"PARTIDA #{self.id:03d}",
            'n_players': len(self.players),
            'creation_time': round(float(self.creation_time), 2),
            'avg_compatibility': round(float(self.avg_compatibility * 100), 1),
            'avg_rank': round(float(self.avg_rank), 2),
            'avg_combat': round(float(self.avg_combat), 2),
            'players': [p.to_dict() for p in self.players]
        }


class MatchmakingSimulator:
    """
    Simulador probabilístico de cola de matchmaking competitivo.
    
    Implementa:
    1. Llegada estocástica de jugadores (proceso de Poisson o pasos discretos).
    2. Asignación de características basada en las distribuciones seleccionadas del dataset.
    3. Algoritmo de compatibilidad por distancia euclidiana normalizada.
    4. Formación de partidas y seguimiento métrico del desempeño del sistema.
    """
    
    def __init__(self, 
                 players_per_match=10, 
                 max_queue_size=100, 
                 compatibility_threshold=0.20, 
                 seed=42,
                 discrete_config=None,
                 continuous_config=None):
        """
        Inicializa el simulador.
        
        Args:
            players_per_match: Cantidad requerida de jugadores por partida (ej. 10 para 5v5).
            max_queue_size: Capacidad máxima de la cola antes de denegar solicitudes.
            compatibility_threshold: Distancia normalizada máxima permitida (ej. 0.20).
            seed: Semilla para reproducibilidad.
            discrete_config: Tupla (nombre_distribución, dict_parámetros, min_val, max_val).
            continuous_config: Tupla (nombre_distribución, dict_parámetros, min_val, max_val).
        """
        self.players_per_match = int(players_per_match)
        self.max_queue_size = int(max_queue_size)
        self.compatibility_threshold = float(compatibility_threshold)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        
        self.discrete_config = discrete_config or ('Poisson', {'lambda': 11.5}, 1, 18)
        self.continuous_config = continuous_config or ('Normal', {'mu': 85.0, 'sigma': 25.0}, 20, 160)
        
        # Estado interno
        self.queue: List[Player] = []
        self.matches: List[Match] = []
        self.current_time: float = 0.0
        self.next_player_id: int = 1
        self.next_match_id: int = 1
        
        # Historial para métricas y gráficos
        self.history_time: List[float] = []
        self.history_queue_size: List[int] = []
        self.history_matches_count: List[int] = []
        self.history_events: List[Dict[str, Any]] = []
        self.wait_times: List[float] = []
        self.compatibility_scores: List[float] = []
        self.total_attempts: int = 0
        
    def generate_player(self, arrival_time: float) -> Player:
        """
        Genera un nuevo jugador muestreando sus atributos a partir de las
        distribuciones estadísticas seleccionadas para variables discretas y continuas.
        """
        d_name, d_params, d_min, d_max = self.discrete_config
        c_name, c_params, c_min, c_max = self.continuous_config
        
        # Muestrear variable discreta (ej. Rango de habilidad)
        try:
            d_val = float(generate_discrete(d_name, d_params, size=1, seed=int(self.rng.integers(1, 1000000)))[0])
        except Exception:
            d_val = float(self.rng.integers(1, 19))
            
        # Muestrear variable continua (ej. Rendimiento de combate o Duración)
        try:
            c_val = float(generate_continuous(c_name, c_params, size=1, seed=int(self.rng.integers(1, 1000000)))[0])
        except Exception:
            c_val = float(self.rng.normal(80.0, 20.0))
            
        # Normalizar a [0, 1] según los rangos esperados
        d_range = max(1.0, float(d_max - d_min))
        c_range = max(1.0, float(c_max - c_min))
        
        norm_d = np.clip((d_val - d_min) / d_range, 0.0, 1.0)
        norm_c = np.clip((c_val - c_min) / c_range, 0.0, 1.0)
        
        player = Player(
            id=self.next_player_id,
            skill_rank=d_val,
            combat_stat=c_val,
            norm_features=np.array([norm_d, norm_c], dtype=float),
            arrival_time=arrival_time
        )
        self.next_player_id += 1
        return player

    def calculate_distance(self, p1: Player, p2: Player) -> float:
        """
        Calcula la distancia euclidiana normalizada entre dos jugadores:
        d(p1, p2) = ||f1 - f2||_2 / sqrt(dim)
        Retorna valor en [0, 1] donde 0 es idéntico y 1 es máxima diferencia.
        """
        diff = p1.norm_features - p2.norm_features
        dist = np.linalg.norm(diff) / np.sqrt(len(diff))
        return float(dist)

    def find_compatible_group(self, candidate: Player) -> Optional[List[Player]]:
        """
        Busca en la cola un conjunto de (players_per_match - 1) jugadores compatibles
        con el candidato para conformar una partida balanceada.
        """
        if len(self.queue) < self.players_per_match - 1:
            return None
            
        # Calcular distancias del candidato a todos los jugadores en cola
        scored = []
        for p in self.queue:
            d = self.calculate_distance(candidate, p)
            if d <= self.compatibility_threshold:
                scored.append((p, d))
                
        # Ordenar de mayor compatibilidad (menor distancia) a menor
        scored.sort(key=lambda item: item[1])
        
        needed = self.players_per_match - 1
        if len(scored) >= needed:
            selected_players = [p for p, _ in scored[:needed]]
            
            # Verificar consistencia cruzada dentro del grupo seleccionado
            all_group = selected_players + [candidate]
            max_dist_inside = 0.0
            total_dist_inside = 0.0
            n_pairs = 0
            
            for i in range(len(all_group)):
                for j in range(i + 1, len(all_group)):
                    pair_dist = self.calculate_distance(all_group[i], all_group[j])
                    total_dist_inside += pair_dist
                    n_pairs += 1
                    if pair_dist > max_dist_inside:
                        max_dist_inside = pair_dist
                        
            # Si la distancia máxima del grupo excede un margen de tolerancia (1.5 * threshold), rechazar
            if max_dist_inside <= self.compatibility_threshold * 1.5:
                return all_group
                
        return None

    def step(self) -> Dict[str, Any]:
        """
        Ejecuta un paso individual de simulación:
        Llega un jugador, intenta emparejarse y actualiza el estado.
        """
        self.total_attempts += 1
        
        # Tiempo de inter-llegada exponencial (Proceso de Poisson para solicitudes)
        inter_arrival = float(self.rng.exponential(scale=2.5))
        self.current_time += inter_arrival
        
        new_player = self.generate_player(self.current_time)
        
        event = {
            'time': round(self.current_time, 2),
            'type': 'arrival',
            'player': new_player.to_dict(),
            'message': f"Jugador #{new_player.id:03d} solicitó partida y entró a la cola."
        }
        
        # Comprobar capacidad de la cola
        if len(self.queue) >= self.max_queue_size:
            event['type'] = 'queue_full'
            event['message'] = f"Cola llena ({self.max_queue_size}). Jugador #{new_player.id:03d} fue rechazado."
            self.history_events.append(event)
            return {'event': event, 'match_created': None}
            
        # Intentar formar partida
        group = self.find_compatible_group(new_player)
        match_created = None
        
        if group is not None:
            # Calcular compatibilidad promedio del grupo
            compat_list = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    compat_list.append(1.0 - self.calculate_distance(group[i], group[j]))
            avg_compat = float(np.mean(compat_list)) if compat_list else 1.0
            
            # Registrar tiempos de espera
            for p in group:
                p.wait_time = max(0.0, self.current_time - p.arrival_time)
                p.match_id = self.next_match_id
                self.wait_times.append(p.wait_time)
                if p in self.queue:
                    self.queue.remove(p)
                    
            avg_rank = float(np.mean([p.skill_rank for p in group]))
            avg_combat = float(np.mean([p.combat_stat for p in group]))
            
            match_created = Match(
                id=self.next_match_id,
                players=group,
                creation_time=self.current_time,
                avg_compatibility=avg_compat,
                avg_rank=avg_rank,
                avg_combat=avg_combat
            )
            self.matches.append(match_created)
            self.next_match_id += 1
            self.compatibility_scores.append(avg_compat)
            
            event['type'] = 'match_found'
            event['match'] = match_created.to_dict()
            event['message'] = (
                f"¡MATCH ENCONTRADO! PARTIDA #{match_created.id:03d} creada con {len(group)} jugadores "
                f"(Compatibilidad: {avg_compat * 100:.1f}%)."
            )
        else:
            self.queue.append(new_player)
            event['type'] = 'queued'
            event['message'] = f"Jugador #{new_player.id:03d} esperando en cola (Tamaño cola: {len(self.queue)})."
            
        self.history_time.append(self.current_time)
        self.history_queue_size.append(len(self.queue))
        self.history_matches_count.append(len(self.matches))
        self.history_events.append(event)
        
        return {
            'event': event,
            'match_created': match_created.to_dict() if match_created else None
        }

    def simulate(self, n_players=100) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa para n_players.
        
        Args:
            n_players: Total de jugadores que solicitan partida.
            
        Returns:
            Diccionario estructurado con métricas, gráficos y detalles de partidas.
        """
        self.reset()
        for _ in range(n_players):
            self.step()
            
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Calcula el resumen estadístico global de la simulación."""
        total_gen = self.next_player_id - 1
        total_matched = sum(len(m.players) for m in self.matches)
        in_queue = len(self.queue)
        n_matches = len(self.matches)
        
        match_rate = (total_matched / total_gen * 100) if total_gen > 0 else 0.0
        avg_wait = float(np.mean(self.wait_times)) if self.wait_times else 0.0
        max_wait = float(np.max(self.wait_times)) if self.wait_times else 0.0
        avg_compat = float(np.mean(self.compatibility_scores) * 100) if self.compatibility_scores else 0.0
        
        return {
            'total_players': int(total_gen),
            'players_matched': int(total_matched),
            'players_in_queue': int(in_queue),
            'matches_created': int(n_matches),
            'match_rate': round(float(match_rate), 2),
            'avg_wait_time': round(float(avg_wait), 2),
            'max_wait_time': round(float(max_wait), 2),
            'avg_compatibility': round(float(avg_compat), 2),
            'total_attempts': int(self.total_attempts),
            'queue_sizes': self.history_queue_size,
            'time_points': [round(t, 2) for t in self.history_time],
            'wait_times': [round(w, 2) for w in self.wait_times],
            'matches_count_timeline': self.history_matches_count,
            'matches': [m.to_dict() for m in self.matches],
            'queued_players': [p.to_dict() for p in self.queue],
            'recent_events': self.history_events[-20:] if self.history_events else []
        }

    def reset(self):
        """Reinicia el simulador conservando la configuración."""
        self.queue = []
        self.matches = []
        self.current_time = 0.0
        self.next_player_id = 1
        self.next_match_id = 1
        self.history_time = []
        self.history_queue_size = []
        self.history_matches_count = []
        self.history_events = []
        self.wait_times = []
        self.compatibility_scores = []
        self.total_attempts = 0
        self.rng = np.random.default_rng(self.seed)
