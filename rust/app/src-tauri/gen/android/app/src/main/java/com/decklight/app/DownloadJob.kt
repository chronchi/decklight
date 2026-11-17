package com.decklight.app

import android.Manifest
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.job.JobInfo
import android.app.job.JobParameters
import android.app.job.JobScheduler
import android.app.job.JobService
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import java.io.File
import org.json.JSONObject

// The one background path for downloads (AGENTS.md "Downloads keep going without the app"). The download itself runs
// in the app's Rust server; this user-initiated data transfer job keeps the process alive while it does, with the app
// closed, and shows it in one notification. The server writes what it is doing to <root>/downloads/status.json
// (rust/decklight-server/src/bgjob.rs) and removes the file when it ends; the job reads it every second and finishes
// once it is gone. The job also writes <root>/downloads/network from the network callback, so Wi-Fi only still
// holds when the screen (and with it the web client) is gone.
//
// The same job keeps an Ask answer going with the app in the background (kind "answer", user 2026-11-16: "go to another
// app while we wait for the answer, we can come back and finish the answer"): Android cuts a backgrounded app's
// network, which failed an answer with a DNS error. The server writes <root>/answer/status.json while it answers and
// <root>/answer/done.json when it ends; with the app off screen the job then posts "Answer ready" (or "failed").
class DownloadJob : JobService() {
  private val main = Handler(Looper.getMainLooper())
  private var params: JobParameters? = null
  private var tick: Runnable? = null
  private var netCb: ConnectivityManager.NetworkCallback? = null
  private var missing = 0
  private var seen = false
  private var kind = DOWNLOADS

  override fun onStartJob(p: JobParameters): Boolean {
    params = p
    kind = p.extras.getString("kind", DOWNLOADS) ?: DOWNLOADS
    channel(this)
    show(p, JSONObject().put("title", if (kind == ANSWER) "" else "a download").put("state", "starting"))
    if (kind == DOWNLOADS) watchNetwork()
    val t = object : Runnable {
      override fun run() {
        val s = status(this@DownloadJob, kind)
        // A short grace at the start only: the server writes the file right after the tap that started the job. Once
        // seen, a missing file means the work ended: finish at once (the grace kept "Answering" up to 15 s after the
        // answer, 2026-11-16).
        if (s == null) {
          if (seen || ++missing >= GRACE_TICKS) { finish(); return }
        } else {
          seen = true
          show(p, s)
        }
        main.postDelayed(this, 1000)
      }
    }
    tick = t
    main.post(t)
    return true
  }

  // The system stopped the job (the user stopped it in Task Manager, or a quota): the server's download keeps going
  // while the process lives; reschedule only if the system asks for it.
  override fun onStopJob(p: JobParameters): Boolean {
    stopWatching()
    return Build.VERSION.SDK_INT >= 34 && p.stopReason != JobParameters.STOP_REASON_USER
  }

  private fun finish() {
    stopWatching()
    if (kind == ANSWER) answered(this)
    params?.let { jobFinished(it, false) }
    params = null
  }

  private fun stopWatching() {
    tick?.let { main.removeCallbacks(it) }
    tick = null
    netCb?.let { cb -> runCatching { cm(this).unregisterNetworkCallback(cb) } }
    netCb = null
    if (kind == DOWNLOADS) File(dir(this), "network").delete()
  }

  private fun watchNetwork() {
    val write = { File(dir(this), "network").apply { parentFile?.mkdirs() }.writeText(networkKind(this)) }
    write()
    val cb = object : ConnectivityManager.NetworkCallback() {
      override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) { write() }
      override fun onLost(network: Network) { write() }
    }
    if (runCatching { cm(this).registerDefaultNetworkCallback(cb) }.isSuccess) netCb = cb
  }

  private fun show(p: JobParameters, s: JSONObject) {
    val n = if (kind == ANSWER) answering(this, s) else notification(this, s)
    val id = if (kind == ANSWER) ANSWER_ID else NOTIFICATION_ID
    if (Build.VERSION.SDK_INT >= 34) {
      setNotification(p, id, n, JOB_END_NOTIFICATION_POLICY_REMOVE)
    } else if (canNotify(this)) {
      (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(id, n)
    }
  }

  companion object {
    private const val JOB_ID = 7101
    private const val NOTIFICATION_ID = 7101
    private const val CHANNEL = "downloads"
    private const val GRACE_TICKS = 15
    private const val ANSWER_ID = 7102
    private const val ANSWER_DONE_ID = 7103
    private const val ANSWERS = "answers"
    private const val ANSWERING = "answering"
    private const val DOWNLOADS = "downloads"
    private const val ANSWER = "answer"

    private fun cm(c: Context) = c.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private fun dir(c: Context) = File(c.dataDir, "root/downloads")

    fun networkKind(c: Context): String = try {
      val caps = cm(c).getNetworkCapabilities(cm(c).activeNetwork)
      when {
        caps == null -> "none"
        caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) -> "wifi"
        Build.VERSION.SDK_INT >= 30 && caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_TEMPORARILY_NOT_METERED) -> "wifi"
        else -> "cellular"
      }
    } catch (e: Exception) {
      "unknown"
    }

    private fun status(c: Context, kind: String): JSONObject? =
      runCatching { JSONObject(File(c.dataDir, "root/$kind/status.json").readText()) }.getOrNull()

    private fun canNotify(c: Context) = Build.VERSION.SDK_INT < 33 ||
      c.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

    private fun channel(c: Context) {
      val nm = c.getSystemService(NOTIFICATION_SERVICE) as NotificationManager
      nm.createNotificationChannel(NotificationChannel(CHANNEL, "Downloads", NotificationManager.IMPORTANCE_LOW).apply {
        description = "Card art and on-device models downloading in the background"
      })
      nm.createNotificationChannel(NotificationChannel(ANSWERING, "Answering", NotificationManager.IMPORTANCE_LOW).apply {
        description = "Ask answering with the app in the background"
      })
      nm.createNotificationChannel(NotificationChannel(ANSWERS, "Answers", NotificationManager.IMPORTANCE_DEFAULT).apply {
        description = "An Ask answer is ready (or failed) while the app was in the background"
      })
    }

    private fun openApp(c: Context) = PendingIntent.getActivity(
      c, 0, Intent(c, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
      PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
    )

    private fun answering(c: Context, s: JSONObject): Notification =
      Notification.Builder(c, ANSWERING)
        .setSmallIcon(android.R.drawable.stat_notify_chat)
        .setContentTitle("Answering")
        .setContentText(s.optString("title").ifEmpty { "Your question" })
        .setProgress(0, 0, true)
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .setContentIntent(openApp(c))
        .build()

    // The answer ended. On screen the user sees it already; off screen one notification says it is ready (or failed,
    // with what to do), and opening it shows the chat.
    private fun answered(c: Context) {
      val f = File(c.dataDir, "root/$ANSWER/done.json")
      val d = runCatching { JSONObject(f.readText()) }.getOrNull() ?: return
      f.delete()
      if (MainActivity.visible || !canNotify(c)) return
      val ok = d.optBoolean("ok", false)
      val n = Notification.Builder(c, ANSWERS)
        .setSmallIcon(android.R.drawable.stat_notify_chat)
        .setContentTitle(if (ok) "Answer ready" else "The answer failed: open Ask to see why and try again")
        .setContentText(d.optString("title"))
        .setAutoCancel(true)
        .setContentIntent(openApp(c))
        .build()
      (c.getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(ANSWER_DONE_ID, n)
    }

    private fun notification(c: Context, s: JSONObject): Notification {
      val title = s.optString("title", "a download")
      val bytes = s.optLong("bytes", 0)
      val total = s.optLong("total", 0)
      val gb = { b: Long -> "%.1f".format(b / 1e9) }
      val text = when (s.optString("state")) {
        "waiting_wifi" -> "Waiting for Wi-Fi · ${gb(bytes)} of ${gb(total)} GB"
        "starting" -> "Starting…"
        else -> "${gb(bytes)} of ${gb(total)} GB"
      }
      val open = openApp(c)
      val pct = if (total > 0) (bytes * 100 / total).toInt() else 0
      return Notification.Builder(c, CHANNEL)
        .setSmallIcon(android.R.drawable.stat_sys_download)
        .setContentTitle("Downloading $title")
        .setContentText(text)
        .setProgress(100, pct, total <= 0)
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .setContentIntent(open)
        .build()
    }

    /** Starts the job (call from a user's tap, while the app is on screen). Returns "" or the error to show. */
    fun start(a: Activity, kind: String = DOWNLOADS): String = try {
      if (Build.VERSION.SDK_INT >= 33 && !canNotify(a)) {
        a.requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 7101)
      }
      val js = a.getSystemService(Context.JOB_SCHEDULER_SERVICE) as JobScheduler
      val id = if (kind == ANSWER) ANSWER_ID else JOB_ID
      if (js.getPendingJob(id) != null) {
        ""
      } else {
        val b = JobInfo.Builder(id, ComponentName(a, DownloadJob::class.java))
          .setRequiredNetworkType(JobInfo.NETWORK_TYPE_ANY)
          .setExtras(android.os.PersistableBundle().apply { putString("kind", kind) })
        // Before Android 14 there are no user-initiated jobs: a plain job, which the system may run later or stop
        // sooner; the download still runs while the app is open.
        if (Build.VERSION.SDK_INT >= 34) b.setUserInitiated(true)
        if (js.schedule(b.build()) == JobScheduler.RESULT_SUCCESS) "" else "Android refused the background job"
      }
    } catch (e: Exception) {
      Log.w("decklight", "download job", e)
      "Android refused the background job: ${e.message ?: e.javaClass.simpleName}"
    }
  }

  // window.DecklightJobs.start(): the web client calls it when the user starts a download.
  // window.DecklightJobs.answer(): when the user sends a question in Ask (it may only start while the app is on screen).
  class Bridge(private val activity: Activity) {
    @JavascriptInterface
    fun start(): String = run(DOWNLOADS)

    @JavascriptInterface
    fun answer(): String = run(ANSWER)

    private fun run(kind: String): String {
      var r = ""
      val done = java.util.concurrent.CountDownLatch(1)
      activity.runOnUiThread { r = start(activity, kind); done.countDown() }
      done.await(3, java.util.concurrent.TimeUnit.SECONDS)
      return r
    }
  }
}
