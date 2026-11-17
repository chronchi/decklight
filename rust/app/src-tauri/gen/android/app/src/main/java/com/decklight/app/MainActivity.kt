package com.decklight.app

import android.content.Context
import android.content.res.Configuration
import android.graphics.Color
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.view.ViewGroup
import android.view.ViewTreeObserver
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : TauriActivity() {
  private var webView: WebView? = null
  private var networkCallback: ConnectivityManager.NetworkCallback? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    // The bars and the window behind the web view take the saved theme's background (web/styles.css --bg), so the
    // start never flashes the other colour; the web client confirms it through DecklightTheme.apply.
    applyBars(resolve(getSharedPreferences(PREFS, MODE_PRIVATE).getString("theme", "dark") ?: "dark"))
    super.onCreate(savedInstanceState)
    window.decorView.setBackgroundColor(barColor(lastTheme))
    // Edge-to-edge is enforced from API 35: keep the web client out of the status bar, the camera cutout and the
    // gesture bar by padding the content view by those insets.
    val content = findViewById<View>(android.R.id.content)
    ViewCompat.setOnApplyWindowInsetsListener(content) { v, insets ->
      val bars = insets.getInsets(
        WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout() or WindowInsetsCompat.Type.ime()
      )
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      WindowInsetsCompat.CONSUMED
    }
    holdSplash(content)
  }

  // Keep the launch splash (the decklight mark) until the web view has loaded its first page, which draws the same
  // mark, so the start never shows an empty frame between the two; never longer than SPLASH_MAX_MS. The splash then
  // fades out (API 31+).
  private fun holdSplash(content: View) {
    val t0 = SystemClock.uptimeMillis()
    content.viewTreeObserver.addOnPreDrawListener(object : ViewTreeObserver.OnPreDrawListener {
      override fun onPreDraw(): Boolean {
        val web = findWebView(content)
        val ready = (web != null && web.progress >= 100) || SystemClock.uptimeMillis() - t0 > SPLASH_MAX_MS
        if (ready) content.viewTreeObserver.removeOnPreDrawListener(this) else content.postInvalidateDelayed(16)
        return ready
      }
    })
    if (Build.VERSION.SDK_INT >= 31) {
      splashScreen.setOnExitAnimationListener { v ->
        v.animate().alpha(0f).setDuration(220).withEndAction { v.remove() }.start()
      }
    }
  }

  // The web client asks DecklightNet.network() for the real network kind: WebView's navigator.connection has only
  // effectiveType, which reports "4g" on good Wi-Fi too. A metered Wi-Fi hotspot counts as cellular, which is what
  // the user's "Wi-Fi only" means (do not spend paid data).
  override fun onWebViewCreate(webView: WebView) {
    super.onWebViewCreate(webView)
    this.webView = webView
    webView.addJavascriptInterface(NetBridge(applicationContext), "DecklightNet")
    webView.addJavascriptInterface(ShareBridge(this), "DecklightShare")
    webView.addJavascriptInterface(DownloadJob.Bridge(this), "DecklightJobs")
    webView.addJavascriptInterface(ThemeBridge(), "DecklightTheme")
    val cb = object : ConnectivityManager.NetworkCallback() {
      override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) = notifyNet()
      override fun onLost(network: Network) = notifyNet()
    }
    try {
      (getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager).registerDefaultNetworkCallback(cb)
      networkCallback = cb
    } catch (e: Exception) {
      // No ACCESS_NETWORK_STATE or no callback support: the bridge still answers on demand.
    }
  }

  // Settings → Appearance (web/app.js applyTheme). The web view's prefers-color-scheme does not follow the phone (it
  // reported light with dark mode on), so system() reads the phone's dark mode here; apply() recolours the bars, keeps
  // the choice for the next start and picks the next launch's splash (API 33+).
  private inner class ThemeBridge {
    @JavascriptInterface
    fun system(): String = systemTheme()

    // The choice saved here, for pages with no localStorage of their own yet (the startup page, a new origin).
    @JavascriptInterface
    fun mode(): String = getSharedPreferences(PREFS, MODE_PRIVATE).getString("theme", "dark") ?: "dark"

    @JavascriptInterface
    fun apply(mode: String, resolved: String) {
      getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString("theme", mode).apply()
      runOnUiThread {
        applyBars(if (resolved == "light") "light" else "dark")
        window.decorView.setBackgroundColor(barColor(lastTheme))
        if (Build.VERSION.SDK_INT >= 33) {
          splashScreen.setSplashScreenTheme(
            when (mode) {
              "light" -> R.style.Theme_decklight_app_SplashLight
              "system" -> R.style.Theme_decklight_app_SplashSystem
              else -> R.style.Theme_decklight_app
            }
          )
        }
      }
    }
  }

  private var lastTheme = "dark"

  private fun systemTheme(): String =
    if ((resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_NO) "light"
    else "dark"

  private fun resolve(mode: String): String = if (mode == "system") systemTheme() else if (mode == "light") "light" else "dark"

  private fun barColor(theme: String): Int = Color.parseColor(if (theme == "light") "#f5f1e8" else "#141310")

  // Dark icons on paper, light icons on night.
  private fun applyBars(theme: String) {
    lastTheme = theme
    val c = barColor(theme)
    val style = if (theme == "light") SystemBarStyle.light(c, c) else SystemBarStyle.dark(c)
    enableEdgeToEdge(style, style)
  }

  // The phone switched dark mode (uiMode is in configChanges, so the activity stays): the System choice follows.
  override fun onConfigurationChanged(newConfig: Configuration) {
    super.onConfigurationChanged(newConfig)
    webView?.evaluateJavascript("window.__decklightUiMode && window.__decklightUiMode()", null)
  }

  private fun notifyNet() {
    webView?.post {
      webView?.evaluateJavascript("window.__decklightNetChanged && window.__decklightNetChanged()", null)
    }
  }

  // DownloadJob posts "Answer ready" only when the app is not on screen.
  override fun onResume() {
    super.onResume()
    visible = true
  }

  override fun onPause() {
    visible = false
    super.onPause()
  }

  override fun onDestroy() {
    networkCallback?.let {
      try {
        (getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager).unregisterNetworkCallback(it)
      } catch (e: Exception) {
        // Already gone.
      }
    }
    networkCallback = null
    webView = null
    super.onDestroy()
  }

  private class NetBridge(private val context: Context) {
    @JavascriptInterface
    fun network(): String = try {
      val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
      val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return "none"
      when {
        caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) -> "wifi"
        Build.VERSION.SDK_INT >= 30 &&
          caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_TEMPORARILY_NOT_METERED) -> "wifi"
        else -> "cellular"
      }
    } catch (e: Exception) {
      "unknown"
    }
  }

  private fun findWebView(v: View): WebView? {
    if (v is WebView) return v
    if (v is ViewGroup) for (i in 0 until v.childCount) findWebView(v.getChildAt(i))?.let { return it }
    return null
  }

  companion object {
    @Volatile var visible = false
    private const val SPLASH_MAX_MS = 2500L
    private const val PREFS = "decklight"
  }
}
