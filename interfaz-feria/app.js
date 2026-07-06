/* ============================================================================
   app.js — Motor de predicción de la interfaz (replica el XGBoost del notebook)
   Requiere window.MODELO_DATA (definido en modelo_web.js).
   ========================================================================== */
(function () {
  "use strict";

  const AÑO_REF = 2026; // igual que el notebook: antigüedad = 2026 - Año
  const D = window.MODELO_DATA;

  // ── Estado UI: modelo real vs. ejemplo ────────────────────────────────
  const bar = document.getElementById("modelbar");
  const esEjemplo = !D || D.__ejemplo === true;
  if (!D) {
    bar.innerHTML = '<span class="warn">⚠ No se encontró el modelo.</span> ' +
      "Exporta <code>modelo_web.js</code> desde Colab y déjalo en esta carpeta.";
  } else if (esEjemplo) {
    bar.innerHTML = '<span class="warn">⚠ Usando modelo de EJEMPLO</span> — ' +
      "reemplaza <code>modelo_web.js</code> por el exportado desde tu notebook para predicciones reales.";
  } else {
    const n = (D.trees || []).length;
    bar.innerHTML = `<span class="ok">✓ Modelo cargado</span> — ${n} árboles · ` +
      `${Object.keys(D.marca_modelos || {}).length} marcas.`;
  }

  // ── Formato CLP ───────────────────────────────────────────────────────
  const clp = (n) =>
    "$" + Math.round(n).toLocaleString("es-CL");

  // ── Target encoding lookup ────────────────────────────────────────────
  function teLookup(campo, valor) {
    const m = (D.te_maps && D.te_maps[campo]) || {};
    if (Object.prototype.hasOwnProperty.call(m, valor)) return m[valor];
    return (D.te_globals && D.te_globals[campo]) || 0;
  }

  // ── Construir el vector de features por NOMBRE (el árbol referencia por nombre) ──
  function construirFeatures(inp) {
    const f = {};
    f["antigüedad_auto"] = AÑO_REF - inp.anio;
    f["Kilometraje"] = inp.km;
    f["Marca_te"] = teLookup("Marca", inp.marca);
    f["Modelo_te"] = teLookup("Modelo", inp.modelo);

    // One-Hot (drop_first): la categoría base queda en 0
    const ohe = D.ohe || {};
    (ohe.Combustible ? ohe.Combustible.cols : []).forEach((c) => {
      f["Combustible_" + c] = inp.combustible === c ? 1 : 0;
    });
    (ohe.Transmision ? ohe.Transmision.cols : []).forEach((c) => {
      f["Transmision_" + c] = inp.transmision === c ? 1 : 0;
    });
    return f;
  }

  // ── Recorrer un árbol XGBoost (formato get_dump JSON) ─────────────────
  //    XGBoost compara internamente en float32 -> Math.fround replica el
  //    enrutamiento exacto (evita errores por precisión en los umbrales).
  function evalArbol(nodo, f) {
    while (!("leaf" in nodo)) {
      const v = f[nodo.split];
      let sig;
      if (v === undefined || v === null || Number.isNaN(v)) {
        sig = nodo.missing;
      } else {
        sig = Math.fround(v) < Math.fround(nodo.split_condition) ? nodo.yes : nodo.no;
      }
      nodo = nodo.children.find((c) => c.nodeid === sig);
      if (!nodo) return 0;
    }
    return nodo.leaf;
  }

  // ── Predicción: base_score + Σ hojas (en float32), luego expm1 ─────────
  //    (el modelo se entrenó con target = log1p(price))
  function predecir(inp) {
    const f = construirFeatures(inp);
    let logPred = Math.fround(D.base_score || 0);
    for (const t of D.trees || []) {
      logPred = Math.fround(logPred + Math.fround(evalArbol(t, f)));
    }
    return Math.expm1(logPred); // vuelve a CLP
  }

  // ══════════════════════════════════════════════════════════════════════
  //  Contribuciones por característica (estilo SHAP — método por camino)
  //  Requiere que el modelo se haya exportado con 'cover' (with_stats=True).
  // ══════════════════════════════════════════════════════════════════════
  let _prepared = false, _tieneCover = false;

  // Valor de un nodo = media de las hojas de su subárbol, ponderada por 'cover'.
  function prepNodo(nodo) {
    if ("leaf" in nodo) {
      nodo.__val = nodo.leaf;
      nodo.__cov = nodo.cover != null ? nodo.cover : 1;
      return;
    }
    let acc = 0, cov = 0;
    for (const c of nodo.children) {
      prepNodo(c);
      acc += c.__val * c.__cov;
      cov += c.__cov;
    }
    nodo.__val = cov ? acc / cov : 0;
    nodo.__cov = nodo.cover != null ? nodo.cover : cov;
  }

  function prepararArboles() {
    if (_prepared) return;
    _prepared = true;
    const t0 = (D.trees || [])[0];
    _tieneCover = !!(t0 && t0.cover != null);
    if (_tieneCover) (D.trees || []).forEach(prepNodo);
  }

  function nombreHumano(feat) {
    const map = {
      "antigüedad_auto": "Antigüedad",
      "Kilometraje": "Kilometraje",
      "Marca_te": "Marca",
      "Modelo_te": "Modelo",
      "Transmision_Manual": "Transmisión",
    };
    if (map[feat]) return map[feat];
    if (feat.indexOf("Combustible_") === 0) return "Combustible";
    return feat;
  }

  // Devuelve { base:<log>, contribs:{nombreHumano: <log>} }
  function contribuciones(inp) {
    prepararArboles();
    if (!_tieneCover) return null;
    const f = construirFeatures(inp);
    let base = D.base_score || 0;
    const raw = {};
    for (const t of D.trees || []) {
      base += t.__val;              // valor esperado del árbol (raíz)
      let nodo = t;
      while (!("leaf" in nodo)) {
        const v = f[nodo.split];
        let sig;
        if (v === undefined || v === null || Number.isNaN(v)) sig = nodo.missing;
        else sig = Math.fround(v) < Math.fround(nodo.split_condition) ? nodo.yes : nodo.no;
        const hijo = nodo.children.find((c) => c.nodeid === sig);
        raw[nodo.split] = (raw[nodo.split] || 0) + (hijo.__val - nodo.__val);
        nodo = hijo;
      }
    }
    const contribs = {};
    for (const k in raw) {
      const h = nombreHumano(k);
      contribs[h] = (contribs[h] || 0) + raw[k];
    }
    return { base, contribs };
  }

  // ── Dibujar el gráfico de contribuciones ──────────────────────────────
  function renderShap(inp, pred) {
    const wrap = document.getElementById("shapWrap");
    if (!wrap) return;
    const res = contribuciones(inp);
    if (!res) { wrap.style.display = "none"; return; } // modelo sin 'cover'
    wrap.style.display = "block";

    const precioBase = Math.expm1(res.base);
    document.getElementById("shapBase").innerHTML =
      `Partiendo de un auto promedio (<b>${clp(precioBase)}</b>), cada característica ` +
      `ajusta el precio hasta el valor predicho (<b>${clp(pred)}</b>):`;

    const items = Object.keys(res.contribs)
      .map((k) => ({ name: k, val: res.contribs[k] }))
      .filter((x) => Math.abs(x.val) > 1e-4)
      .sort((a, b) => Math.abs(b.val) - Math.abs(a.val));

    const maxAbs = items.reduce((m, x) => Math.max(m, Math.abs(x.val)), 1e-6);

    document.getElementById("shap").innerHTML = items.map((x) => {
      const pct = (Math.exp(x.val) - 1) * 100;    // efecto multiplicativo sobre el precio
      const w = (Math.abs(x.val) / maxAbs) * 48;  // ancho (máx 48% del track)
      const pos = x.val > 0;
      const color = pos ? "var(--red)" : "var(--brand-2)";
      const barStyle = (pos ? `left:50%;` : `right:50%;`) + `width:${w}%;background:${color}`;
      const lbl = (pct >= 0 ? "+" : "") + pct.toFixed(1) + "%";
      return `<div class="shap-row">
          <div class="shap-name">${x.name}</div>
          <div class="shap-track"><span class="shap-mid"></span>
            <span class="shap-bar" style="${barStyle}"></span></div>
          <div class="shap-val" style="color:${color}">${lbl}</div>
        </div>`;
    }).join("");
  }

  // ── Poblar desplegables ───────────────────────────────────────────────
  const selMarca = document.getElementById("marca");
  const selModelo = document.getElementById("modelo");
  const selAnio = document.getElementById("anio");

  const marcas = Object.keys(D && D.marca_modelos ? D.marca_modelos : {}).sort();
  marcas.forEach((m) => selMarca.add(new Option(m, m)));

  selMarca.addEventListener("change", () => {
    selModelo.innerHTML = "";
    const mods = (D.marca_modelos && D.marca_modelos[selMarca.value]) || [];
    if (!mods.length) {
      selModelo.add(new Option("Sin modelos", ""));
      selModelo.disabled = true;
      return;
    }
    selModelo.disabled = false;
    selModelo.add(new Option("Selecciona…", ""));
    mods.forEach((m) => selModelo.add(new Option(m, m)));
  });

  // Años
  const stats = (D && D.stats) || { anio_min: 1995, anio_max: 2026 };
  for (let a = stats.anio_max; a >= stats.anio_min; a--) selAnio.add(new Option(a, a));
  selAnio.value = 2018;

  // ── Modo vender / comprar (solo cambia textos) ────────────────────────
  let modo = "vender";
  document.querySelectorAll("#modo button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#modo button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      modo = b.dataset.modo;
      const lbl = modo === "vender" ? "Precio al que lo quiero vender" : "Precio publicado que encontré";
      document.getElementById("precioLabel").innerHTML = lbl + ' <span class="hint">(CLP)</span>';
      document.getElementById("precioIngresadoLabel").textContent =
        modo === "vender" ? "Tu precio" : "Precio publicado";
    });
  });

  // ── Umbrales ──────────────────────────────────────────────────────────
  const U = (D && D.umbrales) || { inf: -12.5, sup: 10.3 };

  // ── Clasificación + mensajes ──────────────────────────────────────────
  function clasificar(diffPct) {
    if (diffPct > U.sup)
      return {
        cls: "v-red", emoji: "🔴", titulo: "Sobrevalorado",
        color: "diff-pos",
        text: modo === "vender"
          ? "Tu precio está por encima del mercado. Podrías tardar más en vender."
          : "El precio publicado está por encima del mercado. Hay margen para negociar.",
      };
    if (diffPct < U.inf)
      return {
        cls: "v-green", emoji: "🟢", titulo: "Subvalorado",
        color: "diff-neg",
        text: modo === "vender"
          ? "Tu precio está por debajo del mercado. Podrías pedir más."
          : "El precio publicado está por debajo del mercado. Puede ser una buena oportunidad.",
      };
    return {
      cls: "v-yellow", emoji: "🟡", titulo: "Precio justo",
      color: "diff-neu",
      text: "El precio está dentro del rango normal de mercado para un auto con estas características.",
    };
  }

  // ── Evaluar ───────────────────────────────────────────────────────────
  document.getElementById("form").addEventListener("submit", (e) => {
    e.preventDefault();

    const inp = {
      marca: selMarca.value,
      modelo: selModelo.value,
      anio: parseInt(selAnio.value, 10),
      km: parseFloat(document.getElementById("km").value),
      combustible: document.getElementById("combustible").value,
      transmision: document.getElementById("transmision").value,
    };
    const precioUser = parseFloat(document.getElementById("precio").value);

    if (!inp.marca || !inp.modelo) return alert("Selecciona marca y modelo.");
    if (!(inp.km >= 0)) return alert("Ingresa un kilometraje válido.");
    if (!(precioUser > 0)) return alert("Ingresa el precio en CLP.");

    const pred = predecir(inp);
    const diffPct = ((precioUser - pred) / pred) * 100;
    const v = clasificar(diffPct);

    // Verdict
    const verdict = document.getElementById("verdict");
    verdict.className = "verdict " + v.cls;
    document.getElementById("vEmoji").textContent = v.emoji;
    document.getElementById("vTitle").textContent = v.titulo;
    document.getElementById("vText").textContent = v.text;

    // Precios
    document.getElementById("outUser").textContent = clp(precioUser);
    document.getElementById("outPred").textContent = clp(pred);
    const outDiff = document.getElementById("outDiff");
    outDiff.textContent = (diffPct >= 0 ? "+" : "") + diffPct.toFixed(1) + "%";
    outDiff.className = "pval " + v.color;

    // Medidor: mapear diffPct (−40%..+40%) a 0..100%
    const pos = Math.max(0, Math.min(100, ((diffPct + 40) / 80) * 100));
    document.getElementById("marker").style.left = pos + "%";

    // Nota metodológica
    document.getElementById("note").innerHTML =
      `El modelo predice el precio de mercado a partir de marca, modelo, antigüedad, ` +
      `kilometraje, combustible y transmisión. Se considera <b>precio justo</b> una ` +
      `diferencia entre <b>${U.inf.toFixed(1)}%</b> y <b>+${U.sup.toFixed(1)}%</b> respecto al ` +
      `precio predicho — umbrales calibrados empíricamente con autos reales de Chileautos ` +
      `(percentiles 25 y 75).` +
      (esEjemplo ? "<br><b>⚠ Estás viendo el modelo de ejemplo; los números no son reales aún.</b>" : "");

    // Gráfico de contribuciones (por qué el modelo predice ese precio)
    renderShap(inp, pred);

    const r = document.getElementById("resultado");
    r.classList.add("show");
    r.scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();
