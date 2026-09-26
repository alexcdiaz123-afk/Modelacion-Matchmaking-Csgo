/**
 * JavaScript principal para el Simulador de Matchmaking CS:GO
 * Rediseño Visual Completo
 */

// Configuración global de Plotly con tema oscuro
const plotlyDarkLayout = {
    paper_bgcolor: '#151F35',
    plot_bgcolor: '#151F35',
    font: {
        color: '#F8FAFC',
        family: 'Inter, sans-serif'
    },
    xaxis: {
        gridcolor: '#2A3856',
        zerolinecolor: '#2A3856'
    },
    yaxis: {
        gridcolor: '#2A3856',
        zerolinecolor: '#2A3856'
    },
    margin: {
        l: 60,
        r: 30,
        t: 40,
        b: 60
    }
};

Plotly.setPlotConfig({
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
});

// Función para mostrar notificaciones
function showNotification(message, type = 'info') {
    const alertClass = {
        'info': 'alert-info',
        'success': 'alert-success',
        'warning': 'alert-warning',
        'danger': 'alert-danger'
    }[type] || 'alert-info';
    
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.maxWidth = '400px';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Función para formatear números
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return parseFloat(num).toFixed(decimals);
}

// Función para realizar peticiones fetch con manejo de errores
async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || `Error ${response.status}`);
        }
        
        return data;
    } catch (error) {
        console.error('Error en petición API:', error);
        showNotification('Error: ' + error.message, 'danger');
        throw error;
    }
}

// Toggle sidebar en móvil
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

// Cerrar sidebar al hacer clic fuera en móvil
document.addEventListener('click', function(event) {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    
    if (window.innerWidth <= 900 && sidebar && sidebar.classList.contains('open')) {
        if (!sidebar.contains(event.target) && !toggleBtn?.contains(event.target)) {
            sidebar.classList.remove('open');
        }
    }
});

// Inicialización al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    // Resaltar elemento activo del menú
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    // Agregar botón de toggle en móvil
    if (window.innerWidth <= 900) {
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'sidebarToggle';
        toggleBtn.className = 'btn btn-primary position-fixed';
        toggleBtn.style.cssText = 'top: 10px; left: 10px; z-index: 1001; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;';
        toggleBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>';
        toggleBtn.onclick = toggleSidebar;
        document.body.appendChild(toggleBtn);
    }
});
