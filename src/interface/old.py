st.markdown("##", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
  <h2 style="color:#000078; margin-bottom:0.3rem;">¿Qué hace el Robo Advisor?</h2>
  <p style="color:#000078; margin:0;">Metodología, control de riesgo y automatización para una cartera diversificada.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
        <h3 style="color:#000078; margin-top:0"> Diversificación</h3>
        <p style="color:#000078; margin-bottom:0">
            Construye una cartera diversificada para reducir riesgo y ajustarse al inversor.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
        <h3 style="color:#000078; margin-top:0;"> Perfil de riesgo</h3>
        <p style="color:#000078; margin-bottom:0;">
            Ajusta la asignación de activos según la tolerancia y capacidad de asumir riesgo del inversor.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
        <h3 style="color:#000078; margin-top:0;"> Seguimiento</h3>
        <p style="color:#000078; margin-bottom:0;">
            Permite hacer seguimiento de la cartera de invesión.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<hr style="
    border: none;
    height: 2px;
    background: linear-gradient(
        to right,
        transparent,
        #73EDFF,
        transparent
    );
    margin: 3rem 0 2rem 0;
">
""", unsafe_allow_html=True)

st.markdown("""
   <div style="text-align:center; margin-top: 0.8rem; margin-bottom: 1.2rem;">
     <h2 style="color:#000078; margin-bottom:0.25rem;">Cómo funciona</h2>
     <p style="color:#000078; margin:0;">Pasos para generar la recomendación.</p>
   </div>
   """, unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
        <h3 style="color:#000078; margin-top:0;">Cuestionario</h3>
        <p style="color:#000078; margin-bottom:0;">
            Preguntas para determinar perfil y restricciones.
        </p>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
        <h3 style="color:#000078; margin-top:0;">Perfil</h3>
        <p style="color:#000078; margin-bottom:0;">
            Determinación del perfil de riesgo del inversor a partir del cuestionario.
        </p>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
    <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
        <h3 style="color:#000078; margin-top:0;">Cartera</h3>
        <p style="color:#000078; margin-bottom:0;">
            Presentación de la cartera realizada
        </p>
    </div>
    """, unsafe_allow_html=True)